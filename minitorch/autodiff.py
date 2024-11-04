from dataclasses import dataclass
from typing import Any, Iterable, Tuple
from collections import defaultdict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    new_vals = list(vals)
    new_vals[arg] = new_vals[arg] + epsilon
    print('new_vals:', new_vals, 'vals', vals)
    return (f(*new_vals) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def _dfs_topological_sort(variable: Variable, history_list: Iterable[Variable], names_list: Iterable[str]):
    inputs = variable.history.inputs

    for input in inputs:
        if input.name not in names_list:
            _dfs_topological_sort(input, history_list, names_list)

    history_list.append(variable)
    names_list.append(variable.name)


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    history_list = []
    names_list = []
    _dfs_topological_sort(variable, history_list, names_list)
    return history_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    top_sort = topological_sort(variable)
    top_sort = reversed(top_sort)

    if deriv is None:
        deriv = 1.0
    derivatives = defaultdict(float)
    derivatives[variable.name] = deriv

    for var in top_sort:
        if var.history.last_fn is None:
            var.accumulate_derivative(derivatives[var.name])
            continue

        chain_rule_res = var.chain_rule(derivatives[var.name])
        for input, der in chain_rule_res:
            derivatives[input.name] += der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
