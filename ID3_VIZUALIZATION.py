from ID3_IMPLEMENTATION import Attribute, Leaf
from graphviz import Digraph
import os


def buildTreeImg(rootNode, dataset_name, type, view, output_folder=r'C:\Users\noaca\Desktop\ID3_Algorithim\DecisionTreeGraphs'):
    dot = exportTree(rootNode)
    svg_path = os.path.join(output_folder, f"{dataset_name}_{type}_ID3_DT")
    dot.render(svg_path, format="svg",  cleanup=True)
    if view:
        dot.view()
    

def exportTree(node, dot=None, parent_name=None, edge_label=""):
    if dot is None:
        dot = Digraph()
    
    if isinstance(node, Attribute):
        node_label = f"{node.attribute}"
    elif isinstance(node, Leaf):
        node_label = f"{node.targetValue}\n{node.count}"
    
    curr_node_name = f"{id(node)}" 
    dot.node(curr_node_name, label=node_label, shape="ellipse" if isinstance(node, Leaf) else "box")
    
    if parent_name is not None:
        dot.edge(parent_name, curr_node_name, label=edge_label)
    if isinstance(node, Attribute):
        for child in node.children:
            edge_label = child.attributeValue
            exportTree(child, dot, curr_node_name, edge_label)
                   
    return dot