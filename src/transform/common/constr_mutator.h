#ifndef TVM_TL_TRANSFORM_COMMON_CONSTR_MUTATOR_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTR_MUTATOR_H_

#include "constraint.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt_functor.h"

namespace tvm::tl {

struct ConstrMutator : public tir::StmtExprMutator {
private:
  using Base = tir::StmtExprMutator;

  struct Guard {
    std::vector<Constr> &constrs;
    ~Guard() { constrs.pop_back(); }
  };

protected:
  template <typename... Args> Guard MakeGuard(const Args... args) {
    constr_stack_.push_back(Constr(args...));
    return Guard{constr_stack_};
  }

public:
  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  PrimExpr VisitIfThenElseExpr(const PrimExpr cond, const PrimExpr true_value,
                               const PrimExpr false_value) {
    PrimExpr new_cond = this->VisitExpr(cond);
    PrimExpr new_true_value;
    {
      auto guard = MakeGuard(new_cond);
      new_true_value = this->VisitExpr(true_value);
    }
    PrimExpr new_false_value;
    {
      auto guard = MakeGuard(tir::Not(new_cond));
      new_false_value = this->VisitExpr(false_value);
    }

    if (new_cond.same_as(cond) && new_true_value.same_as(true_value) &&
        new_false_value.same_as(false_value)) {
      return tir::Select(cond, true_value, false_value);
    }
    return tir::Select(new_cond, new_true_value, new_false_value);
  }

  tir::Stmt VisitStmt_(const tir::LetStmtNode *op) override {
    PrimExpr value = this->VisitExpr(op->value);
    auto guard = MakeGuard(op->var, value);
    tir::Stmt body = this->VisitStmt(op->body);

    if (value.same_as(op->value) && body.same_as(op->body)) {
      return tvm::ffi::GetRef<tir::Stmt>(op);
    }
    return tir::LetStmt(op->var, value, body);
  }

  tir::Stmt VisitStmt_(const tir::AttrStmtNode *op) override {
    // First visit value and body
    PrimExpr value = this->VisitExpr(op->value);
    tir::Stmt body = this->VisitStmt(op->body);

    // Then apply constraints based on attr_key
    if (op->attr_key == tir::attr::tilelang_assume) {
      auto expr = Downcast<PrimExpr>(op->node);
      auto guard = MakeGuard(expr, true);
    } else if (op->attr_key == tir::attr::thread_extent ||
               op->attr_key == tir::attr::virtual_thread) {
      tir::IterVar iv = Downcast<tir::IterVar>(op->node);
      Range dom = Range::FromMinExtent(tir::make_zero(value.dtype()), value);
      auto guard = MakeGuard(iv->var, dom);
    }

    // Check if anything changed
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return tvm::ffi::GetRef<tir::Stmt>(op);
    }

    // Build new AttrStmt
    return tir::AttrStmt(op->node, op->attr_key, value, body);
  }

  tir::Stmt VisitStmt_(const tir::AssertStmtNode *op) override {
    PrimExpr condition = this->VisitExpr(op->condition);
    auto guard = MakeGuard(condition);
    PrimExpr message = this->VisitExpr(op->message);
    tir::Stmt body = this->VisitStmt(op->body);

    if (condition.same_as(op->condition) && message.same_as(op->message) &&
        body.same_as(op->body)) {
      return tvm::ffi::GetRef<tir::Stmt>(op);
    }
    return tir::AssertStmt(condition, message, body);
  }

  tir::Stmt VisitStmt_(const tir::IfThenElseNode *op) override {
    PrimExpr condition = this->VisitExpr(op->condition);
    tir::Stmt then_case;
    {
      auto guard = MakeGuard(condition);
      then_case = this->VisitStmt(op->then_case);
    }

    tir::Stmt else_case;
    if (op->else_case) {
      auto guard = MakeGuard(tir::Not(condition));
      else_case = this->VisitStmt(op->else_case.value());
    }

    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        (!op->else_case || else_case.same_as(op->else_case.value()))) {
      return tvm::ffi::GetRef<tir::Stmt>(op);
    }

    if (op->else_case) {
      return tir::IfThenElse(condition, then_case, else_case);
    }
    return tir::IfThenElse(condition, then_case);
  }

  PrimExpr VisitExpr_(const tir::SelectNode *op) override {
    return VisitIfThenElseExpr(op->condition, op->true_value, op->false_value);
  }

  PrimExpr VisitExpr_(const tir::CallNode *op) override {
    static auto op_if_then_else = Op::Get("tir.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      return VisitIfThenElseExpr(op->args[0], op->args[1], op->args[2]);
    } else {
      return Base::VisitExpr_(op);
    }
  }

  tir::Stmt VisitStmt_(const tir::ForNode *op) override {
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    tir::Stmt body;

    if (op->kind == tir::ForKind::kParallel ||
        op->kind == tir::ForKind::kVectorized) {
      auto guard_1 = MakeGuard(op->loop_var, Range::FromMinExtent(min, extent));
      auto guard_2 = MakeGuard(extent > 0);
      body = this->VisitStmt(op->body);
    } else {
      auto guard_1 = MakeGuard(op->loop_var, Range::FromMinExtent(min, extent));
      auto guard_2 = MakeGuard(extent > 0);
      body = this->VisitStmt(op->body);
    }

    if (min.same_as(op->min) && extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return tvm::ffi::GetRef<tir::Stmt>(op);
    }
    return tir::For(op->loop_var, min, extent, op->kind, body,
                    op->thread_binding, op->annotations);
  }

  tir::Stmt VisitStmt_(const tir::WhileNode *op) override {
    PrimExpr condition = this->VisitExpr(op->condition);
    tir::Stmt body;
    {
      auto guard = MakeGuard(condition);
      body = this->VisitStmt(op->body);
    }

    if (condition.same_as(op->condition) && body.same_as(op->body)) {
      return tvm::ffi::GetRef<tir::Stmt>(op);
    }
    return tir::While(condition, body);
  }

  std::vector<Constr> constr_stack_;
};

} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_CONSTR_MUTATOR_H_
