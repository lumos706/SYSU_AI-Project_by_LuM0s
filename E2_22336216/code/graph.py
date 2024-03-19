import ast

import graphviz


def parse_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        code = file.read()
        return ast.parse(code, filename=file_name)


def generate_graph(node):
    Graph = graphviz.Digraph(format='png')

    def _traverse(Node):
        if isinstance(Node, ast.FunctionDef):
            Graph.node(str(id(Node)), label=Node.name, shape='rectangle')
            for child in ast.walk(Node):
                if isinstance(child, ast.FunctionDef):
                    Graph.edge(str(id(Node)), str(id(child)))
        elif isinstance(Node, ast.Module):
            for child in Node.body:
                _traverse(child)

    _traverse(node)
    return Graph


if __name__ == "__main__":
    filename = 'E2-22336216.py'
    tree = parse_file(filename)
    graph = generate_graph(tree)
    graph.render('graph', view=True)
