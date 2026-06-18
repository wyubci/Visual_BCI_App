import os
import win32com.client as win32


OUT = r"C:\Users\adam\Desktop\基于脑机接口与实时视觉反馈的交互系统_立项报告_视觉重构版.pptx"
IMG_DIR = os.path.abspath("申报书_images")


def rgb(r, g, b):
    return int(r) + int(g) * 256 + int(b) * 65536


def image(name):
    return os.path.join(IMG_DIR, name)


def text(slide, s, x, y, w, h, size=20, color=(255, 255, 255), bold=False, align=1):
    box = slide.Shapes.AddTextbox(1, x, y, w, h)
    tr = box.TextFrame.TextRange
    tr.Text = s
    tr.Font.Name = "Microsoft YaHei"
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = rgb(*color)
    tr.ParagraphFormat.Alignment = align
    box.TextFrame.MarginLeft = 4
    box.TextFrame.MarginRight = 4
    box.TextFrame.MarginTop = 2
    box.TextFrame.MarginBottom = 2
    return box


def rect(slide, x, y, w, h, fill=(14, 45, 82), line=(0, 225, 255), trans=0.08):
    r = slide.Shapes.AddShape(1, x, y, w, h)
    r.Fill.ForeColor.RGB = rgb(*fill)
    r.Fill.Transparency = trans
    r.Line.ForeColor.RGB = rgb(*line)
    r.Line.Transparency = 0.18
    r.Line.Weight = 1.1
    return r


def card(slide, x, y, w, h, title, body, accent=(0, 240, 255), body_size=17):
    rect(slide, x, y, w, h)
    text(slide, title, x + 14, y + 10, w - 28, 28, 22, accent, True)
    text(slide, body, x + 16, y + 46, w - 32, h - 52, body_size, (239, 246, 255))


def bg(slide, prs):
    slide.FollowMasterBackground = False
    slide.Background.Fill.ForeColor.RGB = rgb(3, 21, 54)
    for i, col in enumerate([(0, 70, 118), (0, 120, 170), (0, 220, 245)]):
        band = slide.Shapes.AddShape(1, -60, 420 + i * 26, prs.PageSetup.SlideWidth + 120, 110)
        band.Fill.ForeColor.RGB = rgb(*col)
        band.Fill.Transparency = 0.64 + i * 0.08
        band.Line.Visible = 0
    rail = slide.Shapes.AddLine(36, 28, 36, 510)
    rail.Line.ForeColor.RGB = rgb(0, 240, 255)
    rail.Line.Transparency = 0.35
    rail.Line.Weight = 1.2
    text(slide, "WYU · BCI", 790, 24, 125, 24, 13, (225, 248, 255), True, 2)


def header(slide, section):
    text(slide, "项目内容", 54, 30, 155, 38, 29, (255, 255, 255), True)
    ln = slide.Shapes.AddLine(218, 52, 292, 52)
    ln.Line.ForeColor.RGB = rgb(255, 255, 255)
    ln.Line.Weight = 2
    text(slide, section, 305, 32, 400, 34, 23, (255, 255, 255), True)


def slide(prs, section=None):
    s = prs.Slides.Add(prs.Slides.Count + 1, 12)
    bg(s, prs)
    if section:
        header(s, section)
    return s


def pic_fit(slide, path, x, y, w, h, pad=0, panel=True):
    if panel:
        p = rect(slide, x, y, w, h, fill=(246, 250, 255), line=(0, 220, 245), trans=0)
    img = slide.Shapes.AddPicture(os.path.abspath(path), 0, -1, x + pad, y + pad, w - pad * 2, h - pad * 2)
    try:
        img.LockAspectRatio = -1
    except Exception:
        pass
    img.Width = w - pad * 2
    if img.Height > h - pad * 2:
        img.Height = h - pad * 2
    img.Left = x + (w - img.Width) / 2
    img.Top = y + (h - img.Height) / 2
    return img


def pic_cover(slide, path, x, y, w, h, alpha_panel=True):
    img = slide.Shapes.AddPicture(os.path.abspath(path), 0, -1, x, y, w, h)
    if alpha_panel:
        overlay = slide.Shapes.AddShape(1, x, y, w, h)
        overlay.Fill.ForeColor.RGB = rgb(2, 18, 45)
        overlay.Fill.Transparency = 0.35
        overlay.Line.Visible = 0
    return img


def pill(slide, s, x, y, w, h, color=(0, 240, 255)):
    r = slide.Shapes.AddShape(5, x, y, w, h)
    r.Fill.ForeColor.RGB = rgb(5, 38, 72)
    r.Fill.Transparency = 0.08
    r.Line.ForeColor.RGB = rgb(*color)
    r.Line.Weight = 1.2
    text(slide, s, x + 8, y + 8, w - 16, h - 12, 16, (255, 255, 255), True, 2)


def arrow(slide, x1, y1, x2, y2):
    ln = slide.Shapes.AddLine(x1, y1, x2, y2)
    ln.Line.ForeColor.RGB = rgb(0, 240, 255)
    ln.Line.Weight = 2.2


def build():
    app = win32.Dispatch("KWPP.Application")
    try:
        app.Visible = True
    except Exception:
        pass
    if os.path.exists(OUT):
        try:
            os.remove(OUT)
        except OSError:
            pass

    prs = app.Presentations.Add()
    prs.PageSetup.SlideWidth = 960
    prs.PageSetup.SlideHeight = 540

    # 1. Cover: visual 70 / text 30
    s = slide(prs)
    pic_cover(s, image("image13.png"), 0, 0, 960, 540, True)
    text(s, "基于脑机接口与实时视觉反馈的交互系统", 85, 142, 790, 90, 40, (0, 246, 255), True, 2)
    text(s, "大学生创新训练计划项目立项报告", 235, 248, 490, 40, 25, (255, 255, 255), True, 2)
    text(s, "负责人：张连成   指导教师：王洪涛教授\n电子与信息工程学院 · 计算机科学与技术", 285, 326, 390, 58, 17, (235, 246, 255), False, 2)

    # 2. Overview: balanced, UI screenshot must support "what system is"
    s = slide(prs, "系统定位")
    pic_fit(s, image("image1.png"), 54, 102, 535, 330, panel=True)
    text(s, "这不是单纯算法实验，而是一个可运行的脑控交互系统。", 620, 110, 270, 60, 24, (0, 246, 255), True)
    card(s, 625, 190, 260, 82, "输入", "8通道 EEG + 实时视觉反馈", body_size=17)
    card(s, 625, 288, 260, 82, "处理", "TDCA/FBCCA/CCA 解码 + 门控决策", body_size=17)
    card(s, 625, 386, 260, 82, "输出", "小车 / 设备端 / 交互终端控制", body_size=17)

    # 3. Pain point: custom comparison, image only for the new direction
    s = slide(prs, "背景痛点")
    card(s, 65, 115, 250, 225, "传统 BCI", "固定矩阵\n抽象按钮\n用户需要记忆映射\n真实目标与控制指令割裂", accent=(255, 85, 85), body_size=19)
    arrow(s, 335, 230, 430, 230)
    pic_fit(s, image("image5.jpeg"), 455, 96, 425, 278, pad=6, panel=True)
    text(s, "本项目方案", 600, 390, 135, 28, 24, (0, 246, 255), True, 2)
    text(s, "把频率刺激叠加到摄像头画面中的真实目标上，实现“看见目标，即可选择目标”。",
         465, 430, 405, 45, 19, (255, 255, 255), False, 2)

    # 4. Objectives: diagram-heavy, little text
    s = slide(prs, "研究目标")
    text(s, "围绕“实时、准确、稳定、可复现”四个目标展开", 120, 104, 720, 34, 27, (0, 246, 255), True, 2)
    centers = [(230, 245), (405, 245), (580, 245), (755, 245)]
    labels = [
        ("实时", "短窗识别\n低延迟响应"),
        ("准确", "TDCA判别\n多子带融合"),
        ("稳定", "置信门控\n投票冷却"),
        ("可复现", "日志追踪\n参数留痕"),
    ]
    for (cx, cy), (t, b) in zip(centers, labels):
        oval = s.Shapes.AddShape(9, cx - 58, cy - 58, 116, 116)
        oval.Fill.ForeColor.RGB = rgb(6, 48, 92)
        oval.Fill.Transparency = 0.05
        oval.Line.ForeColor.RGB = rgb(0, 240, 255)
        oval.Line.Weight = 2
        text(s, t, cx - 45, cy - 30, 90, 30, 26, (255, 255, 255), True, 2)
        text(s, b, cx - 58, cy + 8, 116, 42, 15, (220, 245, 255), False, 2)
    for i in range(3):
        arrow(s, centers[i][0] + 65, 245, centers[i + 1][0] - 65, 245)
    text(s, "最终目标：让 BCI 从“离线分类结果”走向“真实任务闭环执行”。", 145, 412, 670, 38, 23, (255, 82, 82), True, 2)

    # 5. Technical route: image3 is exactly the full closed loop.
    s = slide(prs, "总体闭环")
    pic_fit(s, image("image3.png"), 65, 105, 830, 295, pad=8, panel=True)
    pill(s, "状态反馈", 136, 430, 150, 45)
    pill(s, "日志追踪", 330, 430, 150, 45)
    pill(s, "门控决策", 524, 430, 150, 45, (255, 160, 70))
    pill(s, "执行控制", 718, 430, 150, 45)

    # 6. Time alignment: image4 is directly relevant.
    s = slide(prs, "时序对齐")
    pic_fit(s, image("image4.jpeg"), 58, 110, 590, 315, pad=6, panel=True)
    text(s, "为什么要做对齐？", 685, 120, 210, 34, 25, (0, 246, 255), True)
    text(s, "同一刺激若落在不同分析相位，模型看到的样本语义会漂移。\n\n项目统一记录刺激起点、接收窗口、判决时刻与执行时刻，把离线训练和在线测试放到同一时间口径下。",
         680, 178, 225, 210, 19, (245, 250, 255))
    text(s, "解决问题：训练口径 ≈ 在线口径", 680, 410, 225, 30, 20, (255, 82, 82), True, 2)

    # 7. Algorithm: two images, both explain TDCA.
    s = slide(prs, "TDCA解码")
    pic_fit(s, image("image6.jpeg"), 55, 92, 330, 395, pad=6, panel=True)
    pic_fit(s, image("image7.jpeg"), 425, 105, 455, 195, pad=6, panel=True)
    pic_fit(s, image("image8.jpeg"), 425, 318, 455, 150, pad=6, panel=True)
    text(s, "左：完整处理流；右上：判别投影效果；右下：子带融合评分。", 430, 84, 455, 24, 16, (0, 246, 255), True, 2)

    # 8. Decision strategy: gate image plus small KPI block.
    s = slide(prs, "在线决策")
    pic_fit(s, image("image9.jpeg"), 70, 100, 430, 365, pad=8, panel=True)
    text(s, "决策层不直接相信单次分类", 545, 122, 320, 36, 26, (0, 246, 255), True)
    card(s, 545, 184, 145, 112, "置信度", "Top1 - Top2\n衡量候选指令可信度", body_size=15)
    card(s, 715, 184, 145, 112, "连续投票", "多次一致后\n才下发动作", body_size=15)
    card(s, 545, 322, 145, 112, "冷却窗口", "防止一次意图\n重复触发", body_size=15)
    card(s, 715, 322, 145, 112, "闭环指标", "正确率 / 误触发\n漏触发 / 时延", body_size=15)

    # 9. Platform: full hardware relevance, visual heavy.
    s = slide(prs, "实验平台")
    pic_fit(s, image("image13.png"), 55, 100, 535, 320, pad=0, panel=True)
    pic_fit(s, image("image11.png"), 620, 105, 250, 140, pad=5, panel=True)
    pic_fit(s, image("image12.png"), 620, 270, 250, 140, pad=5, panel=True)
    text(s, "平台已具备：受试者实验、EEG采集、小车执行端、刺激界面联调条件。", 95, 448, 770, 30, 21, (255, 255, 255), True, 2)

    # 10. Deliverables: mostly designed, not picture dumping.
    s = slide(prs, "计划与成果")
    text(s, "实施节奏", 75, 112, 150, 30, 27, (0, 246, 255), True)
    steps = [
        ("2026.04-06", "范式设计\n采集联调"),
        ("2026.07-10", "TDCA优化\n参数消融"),
        ("2026.11-2027.03", "动态场景\n闭环实验"),
        ("2027.04-06", "成果整理\n结题答辩"),
    ]
    x0 = 95
    for i, (t, b) in enumerate(steps):
        x = x0 + i * 205
        card(s, x, 170, 160, 118, t, b, body_size=17)
        if i < 3:
            arrow(s, x + 165, 230, x + 200, 230)
    text(s, "预期交付", 75, 340, 150, 30, 27, (0, 246, 255), True)
    pill(s, "创新训练报告", 95, 392, 165, 48)
    pill(s, "软件作品", 300, 392, 145, 48)
    pill(s, "实验数据与日志", 485, 392, 170, 48)
    pill(s, "竞赛 / 论文成果", 695, 392, 170, 48)
    text(s, "汇报完毕  恳请指导", 250, 475, 460, 40, 34, (255, 255, 255), True, 2)

    prs.SaveAs(OUT)
    prs.Close()
    try:
        app.Quit()
    except Exception:
        pass
    print(OUT)


if __name__ == "__main__":
    build()
