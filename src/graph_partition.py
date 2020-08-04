"""graph_partition

    `Greedy Additive Edge Contraction 法 (GAEC)
    <https://arxiv.org/abs/1505.06973>`_ による Edge の階層的な縮約を行う.
      - Graph: Partition 集合 (P) と Edge 集合 (E) の組 (P, E).
      - Partition: Edge の縮約によって結合された Node の集合.
      - Edge: Partition を結ぶ.それぞれに切断コスト w が定義されている.

"""

import heapq
import numpy as np


class Partition:
    """Partition

     Nodeの集合とつながっているEdgeに関する情報を保持.
     各Nodeの集合を定義している.
     グラフの統合の際に、まとめるべきNodeとつなぎ直すEdgeを見つけるために必要となる.

    """

    def __init__(self, nodes):
        """__init__

            自身が持つNodeの集合を定義.

            Args:
                nodes (list): 自身のNodeの集合.中身の型の指定はない.

        """
        self.nodes = nodes
        self.links = {}
        self.merged = False

    def __repr__(self):
        return 'Partition(nodes={})'.format(self.nodes)

    def connect(self, edge):
        """connect

            自身とつながっているPartition及びEdgeをself.linksに格納.

            Args:
                edge (Edge): 自身とつながっているEdge.

        """
        pair = edge.get_pair(self)
        self.links[id(pair)] = (edge, pair)

    def disconnect(self, edge):
        """disconnect

            つながっているedgeを切り離す.

            Args:
                edge (Edge): 自身とつながっているedge.

        """
        pair = edge.get_pair(self)
        self.links.pop(id(pair))

    def merge_nodes(self, partition):
        """merge_nodes

            引数に指定したPartitionを自身のNodeに取り込む.

            Args:
                partition (Partition): 自身とつながっているPartition.

        """
        self.nodes += partition.nodes
        partition.nodes = []
        partition.merged = True

    def get_edge(self, partition):
        """get_edge

            自身と引数で指定したPartitionを繋げているEdgeを返す.

            Args:
                partition (Partition): 自身とつながっているPartition.

        """
        if id(partition) not in self.links:
            return None
        return self.links[id(partition)][0]


class Edge:
    """Edge

        Partition間のEdgeとその重みを保持.
        Partition間の関係を示す.

    """

    def __init__(self, partition0, partition1, weight):
        """__init__

            Edgeの定義.

            Args:
                partition0 (Partition): EdgeにつながっているPartition.
                partition１ (Partition): Edgeにつながっているもう一方のPartition.
                weight (float): Edgeの重み.

        """
        self.pair = (partition0, partition1)
        self.weight = weight
        self.removed = False
        partition0.connect(self)
        partition1.connect(self)

    def __lt__(self, edge):
        # priority que のため逆にしている
        return self.weight > edge.weight

    def __repr__(self):
        return 'Edge({}<=>{})'.format(*self.pair)

    def get_pair(self, partition):
        """get_pair

            Edgeでつながっている二つのPartitionのうち、引数ではないPartitionを返す.

            Args:
                partition (Partition): Edgeにつながっている引数ではないPartition.

        """
        if partition is self.pair[0]:
            return self.pair[1]
        else:
            return self.pair[0]

    def remove(self):
        """remove

            つながっているPartitionを切り、Edgeを消去する.

        """
        partition0, partition1 = self.pair
        partition0.disconnect(self)
        partition1.disconnect(self)
        self.removed = True

    def contract(self):
        """contract

            Partitionを新しくつなげてnew_edgeを作成し、重みを更新する.

            Returns:
                Edge: 新しく作成したEdge.

        """
        partition0, partition1 = self.pair
        new_edge = []

        # edgeが多くつながっているNodeを元Nodeにする.
        if len(partition1.links) > len(partition0.links):
            partition0, partition1 = partition1, partition0

        for edge12, partition2 in list(partition1.links.values()):
            # すでにつながっている場合は何もしない.
            if partition0 is partition2:
                continue
            edge02 = partition0.get_edge(partition2)
            # つながっていない場合は新しくEdgeを作成.
            if edge02 is None:
                edge02 = Edge(partition0, partition2, 0)
                new_edge.append(edge02)

            # それぞれのEdgeの重みを更新.
            edge02.weight += edge12.weight
            edge12.remove()

        partition0.merge_nodes(partition1)
        self.remove()

        return new_edge


def greedy_additive(edges, partitions):
    """greedy_additive

         与えられたEdge(edges)とNodeの集合(partitions)からグラフ分割を行う.

        Args:
           edges (list): Edge.
           partitions (list): Nodeの集合(partition).

        Returns:
           list: グラフ分割後のEdge.
           list: グラフ分割後のNodeの集合(partition).

        Examples:
            >>> p0 = Partition([0])
            >>> p1 = Partition([1])
            >>> p2 = Partition([2])
            >>> p3 = Partition([3])
            >>> e01 = Edge(p0, p1, 1)
            >>> e12 = Edge(p1, p2, 2)
            >>> e23 = Edge(p2, p3, 3)
            >>> e30 = Edge(p3, p0, -10)

            >>> e, p = greedy_additive([e01, e12, e23, e30], [p0, p1, p2, p3])
            >>> e, p
                ([Edge(Partition(nodes=[0])<=>Partition(nodes=[1, 2, 3]))],
                 [Partition(nodes=[0]), Partition(nodes=[1, 2, 3])])
    """

    heapq.heapify(edges)

    while edges:
        edge = heapq.heappop(edges)

        if edge.removed:
            continue

        # 全てのedgeの重みが0以下になったら終了.
        if edge.weight < 0:
            heapq.heappush(edges, edge)
            break

        new_edges = edge.contract()

        for new_edge in new_edges:
            heapq.heappush(edges, new_edge)

    # 結合して不要になったedgeとpartitionを取り除く.
    edges = list(filter(lambda e: not e.removed, edges))
    partitions = list(filter(lambda p: not p.merged, partitions))

    return edges, partitions


def calc_js_div(p_, q_):
    """calc_js_div

         Segmentation Refinementを行うために、
         Jensen-Shannon divergenceの計算を行う.

        Args:
            p_ (ndarray): 片方のピクセルのSegmentationのクラスごとの事後確率.
            q_ (ndarray): もう片方のピクセルのSegmentationのクラスごとの事後確率.

        Returns:
           float: Affinityにかける修正の値.

    """
    p_q_ = (p_+q_)/2+1e-5
    kl_1 = np.sum(p_ * np.log(p_/p_q_ + 1e-5))
    kl_2 = np.sum(q_ * np.log(q_/p_q_ + 1e-5))
    js_d = 0.5 * (kl_1 + kl_2)
    refine = np.exp(-js_d)
    refine = np.clip(refine, 0, 1)

    return refine


def make_ins_seg(outputs, b=0, st_for=0, en_for=5, min_size=5):
    """make_ins_seg

         pixelをNode、AffinityをEdgeの重みとすることで、
         グラフ分割の考え方を用いてInstanceを作成する.
         Partitionクラスはグラフに対する縮約操作と縮約結果で得られた
         同じセグメントと判断される部分グラフを表す.
         Semgentation Refinementを実施する.
         グラフ分割を用いて、出力からInstance segmentation画像と
         instanceごとのpixelのlistを出力.

        Args:
            outputs (tuple): モデルの出力結果.
            b (int, optional): Instance segmentationを作成したい画像のBatch番号.
            st_for (int, optional): 階層構造のどの部分からグラフ分割を始めるかの指定.
            en_for (int, optional): 階層構造のどの部分でグラフ分割を終わるかの指定.
            min_size (int, optional): 最も小さいInstanceのピクセル数を指定.
                これより小さいInstanceは削除される.

        Returns:
           ndarray: Instance segmentation画像.
               行列の形式は(cls数, height, width, 3(RGB))
           list: instanceごとのpixelの位置データのlist.BBox作成に用いる.

    """

    # p: Partition
    # p_list: Partitionに格納されているNodeの各座標
    # e: Edge
    p = []
    p_list = []
    e = []

    # 前層のデータがある位置を示す(0が前層のデータあり).
    pre_detect = np.ones((1, 1))

    for mag in range(st_for, en_for):
        det_segment = outputs[mag].cpu().detach().numpy()[b]
        back = det_segment.shape[0] - 1
        cls_segment = np.argmax(det_segment, axis=0)
        foreground = np.where(cls_segment != back, 1, 0)
        aff = outputs[mag+5].cpu().detach().numpy()[b]

        # 前層から引き継いだInstanceの数.
        pre_node = len(p_list)
        # 前層のデータがない位置のみから新しいNodeを探す.
        foreground = foreground * pre_detect

        # segmentation閾値以上の座標をノードとする.
        for i in range(aff.shape[1]):
            for j in range(aff.shape[2]):
                if foreground[i, j] == 1:
                    p.append(Partition([(i, j)]))
                    p_list.append([(i, j)])

        # 新たなNode同士のEdgeを作成.
        for i in range(pre_node, len(p)):
            for j in range(i+1, len(p)):
                i_y, i_x = p_list[i][0]
                j_y, j_x = p_list[j][0]
                sub_y = j_y - i_y
                sub_x = j_x - i_x
                if (sub_y <= 2 and sub_x >= -2 and sub_x <= 2):
                    # Segmentation Refinement
                    refine = calc_js_div(det_segment[:, i_y, i_x],
                                         det_segment[:, j_y, j_x])
                    ind = 12+sub_x+sub_y*5
                    aff_a = aff[ind, i_y, i_x]
                    aff_b = aff[24-ind, j_y, j_x]
                    aff_ = (aff_a+aff_b)/2
                    aff_ = aff_ * refine
                    aff_ = np.log((aff_+1e-5)/(1-aff_+1e-5))

                    e.append(Edge(p[i], p[j], aff_))

        # 新たなNodeと前層のNodeのEdgeを作成.
        for j in range(pre_node, len(p)):
            for i in range(0, pre_node):
                # 2つのNode間に、すでにEdgeがあるかどうかの目印.
                flag = False
                for pre in p_list[i]:
                    i_y, i_x = pre
                    j_y, j_x = p_list[j][0]
                    sub_y = j_y - i_y
                    sub_x = j_x - i_x
                    if not (sub_y <= 2 and sub_y >= -2
                            and sub_x >= -2 and sub_x <= 2):
                        continue
                    # Segmentation Refinement
                    refine = calc_js_div(det_segment[:, i_y, i_x],
                                         det_segment[:, j_y, j_x])

                    ind = 12+sub_x+sub_y*5
                    aff_a = aff[ind, i_y, i_x]
                    aff_b = aff[24-ind, j_y, j_x]
                    aff_ = (aff_a+aff_b)/2
                    aff_ = aff_ * refine
                    aff_ = np.log((aff_+1e-5)/(1-aff_+1e-5))
                    if flag is False:
                        e.append(Edge(p[i], p[j], aff_))
                        flag = True
                    else:
                        e[-1].weight += aff_

        e, p = greedy_additive(e, p)

        # 最終層時以外は次層に渡すNodeを決める.
        if mag != en_for-1:
            pre_detect = np.ones((aff.shape[1], aff.shape[2]))
            p_list = []
            for i in range(len(p)):
                area_l = p[i].nodes

                # 上下左右が同じグラフであるNodeのみ抽出.
                area_l = [area_l[i] for i in range(len(area_l))
                          if (((area_l[i][0], area_l[i][1]+1) in area_l)
                              and ((area_l[i][0]+1, area_l[i][1]) in area_l)
                              and ((area_l[i][0]-1, area_l[i][1]) in area_l)
                              and ((area_l[i][0], area_l[i][1]-1) in area_l))]
                p_ = []
                # 次層のため、グラフの大きさを縦横2倍に拡大.
                for area_ in area_l:
                    pre_detect[area_[0], area_[1]] = 0
                    p_.append((area_[0]*2, area_[1]*2))
                    p_.append((area_[0]*2+1, area_[1]*2))
                    p_.append((area_[0]*2, area_[1]*2+1))
                    p_.append((area_[0]*2+1, area_[1]*2+1))

                p[i].nodes = p_
                if p_:
                    p_list.append(p_)

            # 次層に引き継がないNode、Edgeを削除.
            p = [p[i] for i in range(len(p)) if p[i].nodes]
            e = [e[i] for i in range(len(e))
                 if ((e[i].pair[0].nodes) and (e[i].pair[1].nodes))]

            pre_detect = pre_detect.repeat(2, axis=0).repeat(2, axis=1)

    # Instance Segmentation画像とNodeのlistを作成.
    ins = np.zeros((det_segment.shape[0],
                    aff.shape[1], aff.shape[2], 3), dtype=int)
    ins_list = [[] for i in range(det_segment.shape[0])]
    pre_color = []
    for area in p:
        pos = sorted(list(area.nodes))
        if len(pos) < min_size:
            continue
        cls_value = np.array([0 for i in range(det_segment.shape[0])])
        # 同じ色を生成しないように
        while(True):
            color = np.random.randint(1, 255, 3)
            if not [i for i in range(len(pre_color))
                    if np.sum(pre_color[i] == color) == 3]:
                pre_color.append(color)
                break

        for i in pos:
            cls_value[cls_segment[i[0], i[1]]] += 1
        cls_num = np.argmax(cls_value)

        ins_list[cls_num].append(pos)
        for i in pos:
            ins[cls_num, i[0], i[1]] = color

    ins = ins.repeat(2**(5-en_for), axis=1).repeat(2**(5-en_for), axis=2)

    return ins, ins_list
