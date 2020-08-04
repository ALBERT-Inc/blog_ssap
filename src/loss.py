import torch


def focal_loss(input_, target, gamma=3.0, small_value=0.1):
    """focal_loss

         Semantic Segmentationのlossを
         `Focal loss <https://arxiv.org/abs/1708.02002>`_ を用いて計算.
         小さな間違いに鈍感になるようにし、背景など領域の大きいクラスのLossの影響が小さく
         なるように学習が進む.

        Args:
            input_ (Tensor): 推測したSemantic Segmentation.
                行列の形式は(n_batch, ch(class数), height, width)
            target (Tensor): 正解Semantic Segmentation.
                行列の形式は(n_batch, ch(class数), height, width)
            gamma (float): focal lossのγの値. 大きいほど小さな間違いに鈍感になる.
            small_value (float, optional): 分母が0にならないようにするための値.

        Returns:
            float: Semantic SegmentationのLossを返す.

    """
    # 正解クラスを抽出
    p_t = torch.max(input_*target, 1)[0]
    loss = - ((torch.ones_like(p_t)-p_t)**gamma
              * torch.log(p_t+small_value))

    return torch.mean(loss)


def l2_loss(input_, t_seg, target, weight):
    """l2_loss

         AffinityのlossをL2_Lossを用いて計算.

        Args:
            input_ (Tensor): 推測したSemantic Segmentation.
                行列の形式は(n_batch, ch(AFF_R**2), height, width)
                なお、AFF_RはAffinityを計算する際のWindow size.
            t_seg (Tensor): 正解Semantic Segmentation.
                行列の形式は(n_batch, ch(class数), height, width)
            target (Tensor): 正解Semantic Segmentation.
                行列の形式は(n_batch, ch(AFF_R**2), height, width)
            weight (list): Loss計算時の重み.
                [Object内の重み, Affinityが1の時のdropout率]を格納.

    """
    # Affinityが1のものを支持された率でdropout.
    # targetとinput_の値を同じにしてlossを0にする.
    rand_1 = torch.rand_like(target) < weight[1]
    rand_1 = rand_1.float()
    ones = torch.ones_like(target)
    drop = torch.where(rand_1 == 1., input_, ones)
    target = target * drop

    # Object内とObject外を分ける.
    t_seg_in = torch.sum(t_seg[:, :-1], 1)[:, None]
    t_seg_out = torch.ones_like(t_seg_in) - t_seg_in

    loss = (input_ - target) ** 2
    # Object内とObject外でlossの重みを変える.
    loss = (loss * t_seg_in * weight[0]) + (loss * t_seg_out)

    return torch.mean(loss)


def calc_loss(outputs, labels, t_aff,
              aff_calc_weight, aff_weight, l_aff_weight):
    """calc_loss

         Semantic SegmentationとAffinityのLossを計算する.

        Args:
            outputs (tuple): Semantic SegmentationとAffinityの推測結果.
            labels (Tensor): 正解Semantic Segmentation.
                行列の形式は(n_batch, ch(class数), height, width)
            t_aff (Tensor): 正解Affinity.
                行列の形式は(n_batch, AFF_R, AFF_R**2, height, width)
                なお、AFF_RはAffinityを計算する際のWindow size.
            aff_calc_weight (list): Affinityのloss計算時の重み.
            aff_weight (float): Affinityの全体のlossの重み.
            l_aff_weight (list): Affinityの解像度毎のlossの重み.

        Returns:
            loss_seg (float): Semantic SegmentationのLoss.
            loss_aff (float): AffinityのLoss.

    """
    # Semantic Segmentationはout_c0~out_c4,
    # Affinityはout_aff0~out_aff4にそれぞれ分けられる.
    out_c0, out_c1, out_c2, out_c3, out_c4, \
        out_aff0, out_aff1, out_aff2, out_aff3, out_aff4 = outputs

    img_size = labels.shape[2]

    # Semantic SegmetnationのLoss
    # 識別結果の画像サイズに合わせ、正解データの大きさも揃える.
    loss_c0 = focal_loss(out_c0, labels[:, :, 0:img_size:16, 0:img_size:16])
    loss_c1 = focal_loss(out_c1, labels[:, :, 0:img_size:8, 0:img_size:8])
    loss_c2 = focal_loss(out_c2, labels[:, :, 0:img_size:4, 0:img_size:4])
    loss_c3 = focal_loss(out_c3, labels[:, :, 0:img_size:2, 0:img_size:2])
    loss_c4 = focal_loss(out_c4, labels)

    # AffinityのLoss
    loss_aff0 = l2_loss(out_aff0,
                        labels[:, :, 0:img_size:16, 0:img_size:16],
                        t_aff[:, 4, :, :out_aff0.shape[2], :out_aff0.shape[3]],
                        aff_calc_weight)
    loss_aff1 = l2_loss(out_aff1,
                        labels[:, :, 0:img_size:8, 0:img_size:8],
                        t_aff[:, 3, :, :out_aff1.shape[2], :out_aff1.shape[3]],
                        aff_calc_weight)
    loss_aff2 = l2_loss(out_aff2,
                        labels[:, :, 0:img_size:4, 0:img_size:4],
                        t_aff[:, 2, :, :out_aff2.shape[2], :out_aff2.shape[3]],
                        aff_calc_weight)
    loss_aff3 = l2_loss(out_aff3,
                        labels[:, :, 0:img_size:2, 0:img_size:2],
                        t_aff[:, 1, :, :out_aff3.shape[2], :out_aff3.shape[3]],
                        aff_calc_weight)
    loss_aff4 = l2_loss(out_aff4, labels,
                        t_aff[:, 0, :, :out_aff4.shape[2], :out_aff4.shape[3]],
                        aff_calc_weight)

    loss_seg = loss_c0 + loss_c1 + loss_c2 + loss_c3 + loss_c4

    loss_aff = (loss_aff0*l_aff_weight[0]
                + loss_aff1*l_aff_weight[1]
                + loss_aff2*l_aff_weight[2]
                + loss_aff3*l_aff_weight[3]
                + loss_aff4*l_aff_weight[4])
    loss_aff = loss_aff * aff_weight

    return loss_seg, loss_aff
