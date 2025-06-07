from typing import List, Union
import ast


class AsyncTransformer(ast.NodeTransformer):
    """Converts all async nodes into their synchronous counterparts."""

    def visit_Await(self, node):
        """Awaits are treated as if the keyword was absent."""
        return self.visit(node.value)

    def visit_AsyncFunctionDef(self, node):
        return self.visit(ast.FunctionDef(**node.__dict__))

    def visit_AsyncFor(self, node):
        return self.visit(ast.For(**node.__dict__))

    def visit_AsyncWith(self, node):
        return self.visit(ast.With(**node.__dict__))


class ChainedFunctionTransformer(ast.NodeTransformer):
    def visit_chain(self, node, depth=1):
        if (
            isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Attribute) and
            isinstance(node.value.func.value, ast.Call)
        ):
            # Node is assignment or return with value like `b.c().d()`
            call_node = node.value
            # If we want to handle nested functions in future, depth needs fixing
            temp_var_id = '__chain_tmp_{}'.format(depth)
            # AST tree is from right to left, so d() is the outer Call and b.c() is the inner Call
            unvisited_inner_call = ast.Assign(
                targets=[ast.Name(id=temp_var_id, ctx=ast.Store())],
                value=call_node.func.value,
            )
            ast.copy_location(unvisited_inner_call, node)
            inner_calls = self.visit_chain(unvisited_inner_call, depth + 1)
            for inner_call_node in inner_calls:
                ast.copy_location(inner_call_node, node)
            outer_call = self.generic_visit(type(node)(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=temp_var_id, ctx=ast.Load()),
                        attr=call_node.func.attr,
                        ctx=ast.Load(),
                    ),
                    args=call_node.args,
                    keywords=call_node.keywords,
                ),
                **{field: value for field, value in ast.iter_fields(node) if field != 'value'}  # e.g. targets
            ))
            ast.copy_location(outer_call, node)
            ast.copy_location(outer_call.value, node)
            ast.copy_location(outer_call.value.func, node)
            return [*inner_calls, outer_call]
        else:
            return [self.generic_visit(node)]

    def visit_Assign(self, node):
        return self.visit_chain(node)

    def visit_Return(self, node):
        return self.visit_chain(node)


class IfExpRewriter(ast.NodeTransformer):
    """Splits IfExp ternary expressions containing complex tests into multiple statements

    Will change

    a if b(c) else d

    into

    a if __if_exp_0 else d

    with Assign nodes in assignments [__if_exp_0 = b(c)]
    """

    def __init__(self, starting_index=0):
        self._temporary_variable_index = starting_index
        self.assignments = []

    def visit_IfExp(self, node):
        # raise Exception('IfExp is currently not allowed')

        if isinstance(node.test, (ast.Name, ast.Attribute)):
            return self.generic_visit(node)
        else:
            temp_var_id = '__if_exp_{}'.format(self._temporary_variable_index)
            self._temporary_variable_index += 1
            assignment_of_test = ast.Assign(
                targets=[ast.Name(id=temp_var_id, ctx=ast.Store())],
                value=self.visit(node.test),
            )
            ast.copy_location(assignment_of_test, node)
            self.assignments.append(assignment_of_test)
            transformed_if_exp = ast.IfExp(
                test=ast.Name(id=temp_var_id, ctx=ast.Load()),
                body=self.visit(node.body),
                orelse=self.visit(node.orelse),
            )
            ast.copy_location(transformed_if_exp, node)
            return transformed_if_exp

    def visit_FunctionDef(self, node):
        return node


class IfExpTransformer(ast.NodeTransformer):
    """Goes through module and function bodies, adding extra Assign nodes due to IfExp expressions."""

    def visit_body(self, nodes):
        new_nodes = []
        count = 0
        for node in nodes:
            rewriter = IfExpRewriter(count)
            possibly_transformed_node = rewriter.visit(node)
            if rewriter.assignments:
                new_nodes.extend(rewriter.assignments)
                count += len(rewriter.assignments)
            new_nodes.append(possibly_transformed_node)
        return new_nodes

    def visit_FunctionDef(self, node):
        transformed = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=self.visit_body(node.body),
            decorator_list=node.decorator_list,
            returns=node.returns
        )
        ast.copy_location(transformed, node)
        return self.generic_visit(transformed)

    def visit_Module(self, node):
        transformed = ast.Module(self.visit_body(node.body))
        ast.copy_location(transformed, node)
        return self.generic_visit(transformed)


class MethodCallTransformer(ast.NodeTransformer):
    def __init__(self):
        self.classes: List[ast.ClassDef] = []
        self.call_nodes = []
        self.method_defs = []

    """Convert method calls into class member calls."""
    def visit_ClassDef(self, node):

        # This assumes there is no nested class definition
        self.classes.append(node)
        self.call_nodes.clear()
        self.method_defs.clear()

        new_node = self.generic_visit(node)

        # NOTE: do not allow self(xxx)
        assert all(len(list(self._get_call_names(n.func))) > 1
                   for n in self.call_nodes)
        method_and_nodes = [ (list(self._get_call_names(n.func))[-2], n)
                             for n in self.call_nodes ]

        for m, n in method_and_nodes:
            if m in self.method_defs:
                self.rewrite_call(n)

        self.classes.pop()

        return new_node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'self':
            bases = self._get_base_classes(self.classes[-1])

            if (('pytorch_lightning.LightningModule' in bases)
                or 'torch.nn.Module' in bases):
                
                node.func = ast.Attribute(
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    end_col_offset=node.end_col_offset,
                    value=node.func,
                    attr='forward',
                    ctx=node.func.ctx,
                )


        if_self = self._find_if_self(node.func)
        if if_self:
            self.call_nodes.append(node)

        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if self.classes:
            self.method_defs.append(node.name)

        return self.generic_visit(node)

    def _get_base_classes(self, node: ast.ClassDef) -> List[str]:
        def to_string(n: Union[ast.Attribute, ast.Name]) -> str:
            if isinstance(n, ast.Attribute):
                return to_string(n.value) + '.' + n.attr

            elif isinstance(n, ast.Name):
                return n.id
            
            else:
                raise Exception(f'Failed to get string from {n}')

        return [to_string(b) for b in node.bases]
        

    def _find_if_self(self, node):
        if isinstance(node, ast.Name) and node.id == 'self':
            return True
        elif isinstance(node, ast.Subscript):
            return self._find_if_self(node.value)
        elif isinstance(node, ast.Attribute):
            return self._find_if_self(node.value)
        else:
            return False

    def _get_call_names(self, node):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Subscript):
            yield from self._get_call_names(node.value)
        elif isinstance(node, ast.Str):
            yield node.s
        elif isinstance(node, ast.Attribute):
            yield node.attr
            yield from self._get_call_names(node.value)

    def rewrite_call(self, node):
        self._rewrite_self(node.func)
        self._rewrite_args(node.args)

    def _rewrite_self(self, node):
        if isinstance(node, ast.Name) and node.id == 'self':
            node.id = self.classes[-1].name
        elif isinstance(node, ast.Subscript):
            self._rewrite_self(node.value)
        elif isinstance(node, ast.Attribute):
            self._rewrite_self(node.value)

    def _rewrite_args(self, args):
        args.insert(0, ast.Name(id='self', ctx=ast.Store()))


class PytTransformer(ast.NodeTransformer):
    _transformers = [
        AsyncTransformer,
        ChainedFunctionTransformer,
        IfExpTransformer,
        MethodCallTransformer
    ]

    def __init__(self):
        self.transformers = [ t() for t in self._transformers ]


    def visit(self, node):
        for t in self.transformers:
            node = t.visit(node)

        return node
