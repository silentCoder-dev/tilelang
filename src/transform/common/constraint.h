#ifndef TVM_TL_TRANSFORM_COMMON_CONSTRAINT_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTRAINT_H_

#include "tvm/arith/analyzer.h"
#include "tvm/ffi/base_details.h"
#include "tvm/ffi/object.h"
#include "tvm/ir/expr.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/var.h"
#include <ostream>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm::tl {

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  tir::Var var;
  PrimExpr value;
  Range range;

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(tir::Var var, PrimExpr val)
      : kind(kBindValue), var(var), value(val) {};
  Constr(tir::Var var, Range range)
      : kind(kBindRange), var(var), range(range) {};

  Constr() = default;
  Constr(const Constr &other) = default;
  Constr(Constr &&other) = default;
  Constr &operator=(const Constr &other) = default;

  void format(std::ostream &os) const {
    os << "Constr(kind=";
    switch (kind) {
    case kConstr:
      os << "kConstr";
      os << ", is_assume=" << (is_assume ? "true" : "false");
      os << ", value=" << value;
      break;
    case kBindValue:
      os << "kBindValue";
      os << ", var=" << var->name_hint;
      os << ", value=" << value;
      break;
    case kBindRange:
      os << "kBindRange";
      os << ", var=" << var->name_hint;
      os << ", range=Range(min=" << range->min;
      os << ", extent=" << range->extent << ")";
      break;
    default:
      os << "Unknown";
    }
    os << ")";
  }

  PrimExpr ToGenericConstr() const {
    switch (kind) {
    case kConstr:
      return value;
    case kBindValue:
      return var == value;
    case kBindRange:
      return tir::And(var >= range->min, var < (range->min + range->extent));
    }
    LOG(FATAL) << "Unreachable";
    return PrimExpr();
  }
  Constr Substitute(ffi::Map<tir::Var, PrimExpr> subs) const {
    return Constr(tir::Substitute(ToGenericConstr(), subs));
  }
  void Populate(arith::Analyzer &analyzer) const {
    switch (kind) {
    case kConstr:
      analyzer.EnterConstraint(value);
      break;
    case kBindValue:
      analyzer.Bind(var, value);
      break;
    case kBindRange:
      analyzer.Bind(var, range);
      break;
    default:
      LOG(FATAL) << "Unreachable";
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(ffi::Map<tir::Var, PrimExpr> subs) const {
    ConstrSet new_set;
    for (const auto &c : constrs_) {
      new_set.constrs_.push_back(c.Substitute(subs));
    }
    return new_set;
  }
  void Populate(arith::Analyzer &analyzer) const {
    for (const auto &c : constrs_) {
      c.Populate(analyzer);
    }
  }
  bool CanProve(const PrimExpr &expr) const {
    arith::Analyzer analyzer;
    Populate(analyzer);
    return analyzer.CanProve(expr);
  }
  template <typename... Args> void AddConstr(Args... args) {
    constrs_.push_back(Constr(args...));
  }
  void Extend(const ConstrSet &other) {
    for (const auto &c : other.constrs_) {
      constrs_.push_back(c);
    }
  }

  void format(std::ostream &os) const {
    os << "ConstrSet(size=" << constrs_.size() << ") {\n";
    for (size_t i = 0; i < constrs_.size(); ++i) {
      os << "  [" << i << "] ";
      constrs_[i].format(os);
      os << "\n";
    }
    os << "}";
  }

  std::vector<Constr> constrs_;
};
} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_CONSTRAINT_H_
