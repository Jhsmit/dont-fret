import ast
from functools import reduce
from operator import and_, or_
from typing import Dict

import polars as pl
import yaml

from dont_fret.config.config import Channel


def from_channel(channel: Channel) -> pl.Expr:
    expr = pl.col(channel.target)
    if channel.modulo is not None:
        expr = expr.mod(channel.modulo)
    if isinstance(channel.value, (int, float)):
        expr = expr == pl.lit(channel.value)
    elif len(channel.value) == 2:
        expr = expr.is_between(channel.value[0], channel.value[1], closed="left")
    else:
        raise ValueError(
            "Channel specification field 'value' must be either one value or a range of two"
        )
    return expr


def reduce_and(exprs: list[pl.Expr]) -> pl.Expr:
    return reduce(and_, exprs)


def reduce_or(exprs: list[pl.Expr]) -> pl.Expr:
    return reduce(or_, exprs)


def parse_yaml_expressions(yaml_content: str) -> Dict[str, pl.Expr]:
    yaml_data = yaml.safe_load(yaml_content)
    return {key: parse_expression(value).alias(key) for key, value in yaml_data.items()}


def parse_expression(expr: str) -> pl.Expr:
    tree = ast.parse(expr, mode="eval")
    return evaluate_node(tree.body)


def evaluate_node(node):
    if isinstance(node, ast.Name):
        return pl.col(node.id)
    elif isinstance(node, ast.Constant):
        return pl.lit(node.n)
    elif isinstance(node, ast.BinOp):
        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        return apply_operator(node.op, left, right)
    else:
        raise ValueError(f"Unsupported node type: {type(node).__name__}")


def apply_operator(op, left, right):
    if isinstance(op, ast.Add):
        return left + right
    elif isinstance(op, ast.Sub):
        return left - right
    elif isinstance(op, ast.Mult):
        return left * right
    elif isinstance(op, ast.Div):
        return left / right
    elif isinstance(op, ast.Pow):
        return left.pow(right)
    else:
        raise ValueError(f"Unsupported operator: {type(op).__name__}")
