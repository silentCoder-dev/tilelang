#ifndef TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_

#include "constraint.h"

namespace tvm::tl {

struct ConstrVisitor : public tir::StmtExprVisitor {
private:
  using Base = tir::StmtExprVisitor;

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
  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;
  void VisitIfThenElseExpr(const PrimExpr cond, const PrimExpr true_value,
                           const PrimExpr false_value) {
    {
      auto guard = MakeGuard(cond);
      Base::VisitExpr(true_value);
    }
    {
      auto guard = MakeGuard(tir::Not(cond));
      Base::VisitExpr(false_value);
    }
  }
  void VisitStmt_(const tir::LetStmtNode *op) override {
    auto guard = MakeGuard(op->var, op->value);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const tir::AttrStmtNode *op) override {
    if (op->attr_key == tir::attr::tilelang_assume) {
      auto expr = Downcast<PrimExpr>(op->node);
      auto guard = MakeGuard(expr, true);
      Base::VisitStmt_(op);
    } else if (op->attr_key == tir::attr::thread_extent ||
               op->attr_key == tir::attr::virtual_thread) {
      tir::IterVar iv = Downcast<tir::IterVar>(op->node);
      Range dom =
          Range::FromMinExtent(tir::make_zero(op->value.dtype()), op->value);
      auto guard = MakeGuard(iv->var, dom);
      Base::VisitStmt_(op);
    } else {
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const tir::AssertStmtNode *op) override {
    auto guard = MakeGuard(op->condition);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const tir::IfThenElseNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      auto guard = MakeGuard(tir::Not(op->condition));
      Base::VisitStmt(op->else_case.value());
    }
  }
  void VisitExpr_(const tir::SelectNode *op) override {
    VisitIfThenElseExpr(op->condition, op->true_value, op->false_value);
  }
  void VisitExpr_(const tir::CallNode *op) override {
    static auto op_if_then_else = Op::Get("tir.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      VisitIfThenElseExpr(op->args[0], op->args[1], op->args[2]);
    } else {
      Base::VisitExpr_(op);
    }
  }
  void VisitStmt_(const tir::ForNode *op) override {
    if (op->kind == tir::ForKind::kParallel ||
        op->kind == tir::ForKind::kVectorized) {
      auto guard_1 =
          MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto guard_2 = MakeGuard(op->extent > 0);
      Base::VisitStmt_(op);
    } else {
      auto guard_1 =
          MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto guard_2 = MakeGuard(op->extent > 0);
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const tir::WhileNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->body);
    }
  }
  std::vector<Constr> constr_stack_;
};
} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
