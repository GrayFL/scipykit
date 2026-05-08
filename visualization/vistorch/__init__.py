import torch
from torch import nn
from torch import Tensor
from torch.fx import symbolic_trace, GraphModule

import pandas as pd


def summary_model_pipeline(model: nn.Module):
    """
    对给定的 PyTorch 模型进行符号跟踪，并生成一个包含模型图节点信息的 DataFrame。

    Parameters
    ----------
    model : nn.Module
        需要进行分析的 PyTorch 模型。

    Returns
    -------
    pd.DataFrame
        包含模型图中每个节点的详细信息的 DataFrame，列包括 'opcode', 'name', 'target', 'module_type', 'shape', 'args', 'kwds'。

    Notes
    -----
    此函数使用 `torch.fx.symbolic_trace` 对输入的模型进行符号跟踪，将其转换为 `GraphModule`。
    然后遍历 `GraphModule` 中的每个节点，提取节点的操作码、名称、目标、模块类型、形状、参数和关键字参数等信息。
    对于类型为 'call_module' 的节点，还会获取子模块的类型和权重形状。
    最后，将所有节点信息存储在一个 DataFrame 中，并将缺失值填充为 '-'。

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
    >>> df = summary_model_pipeline(model)
    >>> print(df)
    """
    gm: GraphModule = symbolic_trace(model)
    df_proc = pd.DataFrame( #
        columns=['opcode', 'name', 'target', 'module_type', 'shape', 'args', 'kwds']
        )
    for _i, node in enumerate(gm.graph.nodes):
        # module_type = ""
        node_dic: dict[str, str] = {}
        node_dic['opcode'] = node.op
        node_dic['name'] = node.name
        node_dic['target'] = node.target
        node_dic['args'] = node.args
        node_dic['kwds'] = node.kwargs
        if node.op == 'call_module':
            # 获取子模块实例
            submodule = gm.get_submodule(node.target)
            module_type = submodule.__class__.__name__
            node_dic['module_type'] = module_type
            if 'weight' in submodule.state_dict():
                layer_shape = submodule.state_dict()['weight'].shape
                node_dic['shape'] = layer_shape
        df_proc.loc[_i] = node_dic
    df_proc.fillna('-', inplace=True)
    return df_proc
