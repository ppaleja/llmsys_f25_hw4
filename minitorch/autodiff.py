from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


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
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # BEGIN ASSIGN1_1
    # TODO

    # Hint 1: Ensure the you visit the computation graph in a post-order depth-first search

    # variables is the right-most variable
    # We also know that we only have access to the "parents" of a node.
    # So we need to 
    
    # Topo sort general algorithm:
    # 1. Identify all nodes with no incoming edges (i.e., nodes with no parents).
    # 2. Add that node to the ordering
    # 3. Remove that node and its outgoing edges from the graph.
    # 4. Repeat until all nodes are processed.
    
    # track visited nodes by their unique id
    visited = {}
    res = deque()
    #q = deque([variable])
    
    def explore(v):
        # use .unique_id property (not callable) and guard with get()
        if visited.get(v.unique_id, False):
            return
        visited[v.unique_id] = True
        # If this is a leaf or constant, it has no parents we can traverse.
        # The .parents property may assert when history is None, so guard it.
        if not v.is_leaf() and not v.is_constant():
            for parent in v.parents:
                explore(parent)

        # Hint 2: When the children nodes of the current node are visited,
        # add the current node at the front of the result order list
        if not v.is_constant():
            res.appendleft(v)

    explore(variable)
    return res
            
    # END ASSIGN1_1


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # BEGIN ASSIGN1_1
    # TODO
    ordering = list(topological_sort(variable))  # expected output -> inputs
    # keep our own map of accumulated derivatives so we can pass them to chain_rule
    # unique_id is a @property, access without calling
    grads = {variable.unique_id: deriv}
    for v in ordering:
        # process this node only if we have an accumulated gradient for it
        if v.unique_id not in grads:
            continue
        # pop the gradient for this node so we process it exactly once
        d_out = grads.pop(v.unique_id)
        # If this is a leaf, accumulate the derivative here.
        if v.is_leaf():
            v.accumulate_derivative(d_out)
            continue
        # Constants have no parents and do not accumulate gradients.
        if v.is_constant():
            continue
        # Otherwise, propagate to parents using the chain rule.
        for parent, d_parent in v.chain_rule(d_out):
            grads[parent.unique_id] = grads.get(parent.unique_id, 0) + d_parent
            # only accumulate on parent when we actually reach it (if it's a leaf)
            # the parent.accumulate_derivative will be called when parent is processed
        
   
    # END ASSIGN1_1


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