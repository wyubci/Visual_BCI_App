import os
import win32com.client as win32


OUT = r"C:\Users\adam\Desktop\基于脑机接口与实时视觉反馈的交互系统_立项报告_优化版.pptx"
IMG_DIR = os.path.abspath("申报书_images")


def rgb(r, g, b):
    return int(r) + int(g) * 256 + int(b) * 65536


def img(name):
    return os.path.join(IMG_DIR, name)


def add_text(slide, content, x, y, w, h, size=20, color=(255, 255, 255), bold=False, align=1):
    box = slide.Shapes.AddTextbox(1, x, y, w, h)
    box.TextFrame.MarginLeft = 6
    box.TextFrame.MarginRight = 6
    box.TextFrame.MarginTop = 3
    box.TextFrame.MarginBottom = 3
    tr = box.TextFrame.TextRange
    tr.Text = content
    tr.Font.Name = "Microsoft YaHei"
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = rgb(*color)
    tr.ParagraphFormat.Alignment = align
    return box


def add_rect(slide, x, y, w, h, fill=(12, 45, 82), line=(0, 232, 255), transparency=0.08):
    shape = slide.Shapes.AddShape(1, x, y, w, h)
    shape.Fill.ForeColor.RGB = rgb(*fill)
    shape.Fill.Transparency = transparency
    shape.Line.ForeColor.RGB = rgb(*line)
    shape.Line.Transparency = 0.15
    shape.Line.Weight = 1.2
    return shape


def add_card(slide, x, y, w, h, title, body, accent=(0, 240, 255), body_size=16):
    add_rect(slide, x, y, w, h)
    add_text(slide, title, x + 12, y + 10, w - 24, 28, 21, accent, True)
    add_text(slide, body, x + 14, y + 46, w - 28, h - 54, body_size, (238, 246, 255))


def add_bg(slide, prs):
    slide.FollowMasterBackground = False
    slide.Background.Fill.ForeColor.RGB = rgb(3, 20, 52)

    # bottom technology wave, close to the reference PPT but cleaner
    for i, col in enumerate([(0, 58, 105), (0, 101, 145), (0, 210, 235)]):
        band = slide.Shapes.AddShape(1, -70, 416 + i * 27, prs.PageSetup.SlideWidth + 140, 120)
        band.Fill.ForeColor.RGB = rgb(*col)
        band.Fill.Transparency = 0.62 + i * 0.08
        band.Line.Visible = 0

    rail = slide.Shapes.AddLine(36, 28, 36, 508)
    rail.Line.ForeColor.RGB = rgb(0, 240, 255)
    rail.Line.Transparency = 0.30
    rail.Line.Weight = 1.2

    add_text(slide, "WYU · BCI", 795, 25, 120, 24, 13, (228, 248, 255), True, 2)


def add_header(slide, section, page_no=None):
    add_text(slide, "项目内容", 54, 31, 155, 38, 29, (255, 255, 255), True)
    line = slide.Shapes.AddLine(218, 54, 292, 54)
    line.Line.ForeColor.RGB = rgb(255, 255, 255)
    line.Line.Weight = 2.1
    add_text(slide, section, 306, 34, 390, 32, 23, (255, 255, 255), True)
    if page_no:
        add_text(slide, f"{page_no:02d}", 870, 470, 48, 30, 18, (0, 240, 255), True, 2)


def new_slide(prs, section=None, page_no=None):
    s = prs.Slides.Add(prs.Slides.Count + 1, 12)
    add_bg(s, prs)
    if section:
        add_header(s, section, page_no)
    return s


def add_picture_fit(slide, path, x, y, w, h, pad=6, panel=True):
    if panel:
        add_rect(slide, x, y, w, h, fill=(246, 250, 255), line=(0, 224, 255), transparency=0)
    pic = slide.Shapes.AddPicture(os.path.abspath(path), 0, -1, x + pad, y + pad, w - 2 * pad, h - 2 * pad)
    try:
        pic.LockAspectRatio = -1
    except Exception:
        pass
    pic.Width = w - 2 * pad
    if pic.Height > h - 2 * pad:
        pic.Height = h - 2 * pad
    pic.Left = x + (w - pic.Width) / 2
    pic.Top = y + (h - pic.Height) / 2
    return pic


def add_picture_cover(slide, path, x, y, w, h, overlay=0.0):
    pic = slide.Shapes.AddPicture(os.path.abspath(path), 0, -1, x, y, w, h)
    if overlay > 0:
        o = slide.Shapes.AddShape(1, x, y, w, h)
        o.Fill.ForeColor.RGB = rgb(2, 18, 45)
        o.Fill.Transparency = overlay
        o.Line.Visible = 0
    return pic


def add_bullets(slide, items, x, y, w, size=19, color=(245, 250, 255), gap=34):
    cy = y
    for item in items:
        add_text(slide, "•", x, cy, 22, 26, size + 2, (0, 240, 255), True)
        add_text(slide, item, x + 26, cy, w - 26, 30, size, color)
        cy += gap


def add_tag(slide, label, x, y, w, h, color=(0, 240, 255)):
    r = slide.Shapes.AddShape(5, x, y, w, h)
    r.Fill.ForeColor.RGB = rgb(4, 38, 72)
    r.Fill.Transparency = 0.08
    r.Line.ForeColor.RGB = rgb(*color)
    r.Line.Weight = 1.2
    add_text(slide, label, x + 8, y + 6, w - 16, h - 10, 16, (255, 255, 255), True, 2)


def add_arrow(slide, x1, y1, x2, y2, color=(0, 240, 255)):
    line = slide.Shapes.AddLine(x1, y1, x2, y2)
    line.Line.ForeColor.RGB = rgb(*color)
    line.Line.Weight = 2.2


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

    # 1 cover: reference PPT style, not photo-heavy
    s = new_slide(prs)
    add_text(s, "基于脑机接口与实时视觉反馈的", 110, 160, 740, 58, 42, (0, 246, 255), True, 2)
    add_text(s, "交互系统", 310, 226, 340, 58, 42, (0, 246, 255), True, 2)
    add_text(s, "大学生创新训练计划项目立项报告", 250, 315, 460, 36, 24, (255, 255, 255), True, 2)
    add_text(s, "负责人：张连成    指导教师：王洪涛教授\n电子与信息工程学院 · 计算机科学与技术", 280, 375, 400, 54, 16, (232, 244, 255), False, 2)

    # 2 contents
    s = new_slide(prs, "汇报提纲", 2)
    sections = [
        ("01", "项目背景与需求", "为什么要做动态视觉反馈 BCI"),
        ("02", "研究目标与意义", "从离线识别走向闭环执行"),
        ("03", "研究内容与关键问题", "时序、样本、算法、决策、评估"),
        ("04", "实施方案与技术路线", "动态刺激、TDCA 解码、门控控制"),
        ("05", "基础条件与预期成果", "平台、计划、交付物"),
    ]
    for i, (no, title, desc) in enumerate(sections):
        y = 116 + i * 68
        add_text(s, no, 120, y, 60, 34, 27, (0, 246, 255), True, 2)
        add_text(s, title, 205, y, 240, 32, 24, (255, 255, 255), True)
        add_text(s, desc, 470, y + 4, 360, 28, 17, (220, 236, 250))
        add_arrow(s, 170, y + 17, 198, y + 17)

    # 3 background
    s = new_slide(prs, "项目背景", 3)
    add_text(s, "BCI 技术的现实瓶颈：抽象刺激界面难以承载真实任务", 70, 105, 800, 34, 26, (0, 246, 255), True)
    add_card(s, 78, 160, 250, 210, "传统 SSVEP 界面", "固定位置闪烁块\n频率-指令抽象映射\n用户需记忆目标对应关系\n真实目标与控制动作割裂", accent=(255, 85, 85), body_size=18)
    add_arrow(s, 355, 258, 442, 258)
    add_card(s, 470, 160, 330, 210, "本项目切入点", "摄像头采集真实场景\n目标检测锁定候选物体\n在目标边界叠加频率闪烁\n用户注视真实目标即可选择", body_size=18)
    add_picture_fit(s, img("image5.jpeg"), 90, 395, 700, 95, pad=3)
    add_text(s, "核心转变：从“看按钮”到“看目标”", 650, 394, 240, 34, 22, (255, 80, 80), True, 2)

    # 4 purpose and significance
    s = new_slide(prs, "目的意义", 4)
    add_picture_fit(s, img("image13.png"), 63, 105, 435, 290, pad=0)
    add_card(s, 530, 108, 330, 76, "理论意义", "研究 TDCA 在动态视觉场景中的迁移规律与失效模式。", body_size=16)
    add_card(s, 530, 203, 330, 76, "工程意义", "打通“采集-解码-决策-执行-反馈”的在线闭环。", body_size=16)
    add_card(s, 530, 298, 330, 76, "应用意义", "服务助残康复、远程作业、智能座舱与自然人机交互。", body_size=16)
    add_text(s, "项目目标：实现低延迟、低误触发、可复盘的实时脑控系统。", 120, 430, 720, 36, 23, (0, 246, 255), True, 2)

    # 5 research contents
    s = new_slide(prs, "研究内容", 5)
    items = [
        ("链路一致性与时序对齐", "统一刺激起点、接收窗口、分析窗口和执行时刻"),
        ("预处理与样本治理", "去直流、标准化、异常截断、无效窗剔除"),
        ("TDCA 短窗解码", "时间延迟扩展、判别投影、模板相关、子带融合"),
        ("在线门控与闭环评估", "置信度、连续投票、冷却控制、执行级指标"),
    ]
    for i, (title, body) in enumerate(items):
        add_card(s, 76 + (i % 2) * 410, 122 + (i // 2) * 142, 360, 108, title, body, body_size=17)
    add_text(s, "研究主线：让训练样本、在线样本、执行反馈在同一技术口径下闭合。", 105, 430, 750, 34, 22, (255, 255, 255), True, 2)

    # 6 key problems
    s = new_slide(prs, "关键问题", 6)
    add_picture_fit(s, img("image4.jpeg"), 65, 115, 520, 275, pad=5)
    add_text(s, "拟解决的五个问题", 635, 115, 235, 32, 25, (0, 246, 255), True)
    add_bullets(s, [
        "时序错位导致样本不可分",
        "短窗信息不足与噪声放大",
        "误触发与漏触发并存",
        "训练口径与在线口径不一致",
        "工程链路耦合过高",
    ], 635, 168, 255, size=17, gap=42)
    add_text(s, "解决抓手：统一时间戳 + 有效窗治理 + 门控决策", 145, 426, 670, 34, 22, (255, 80, 80), True, 2)

    # 7 implementation route
    s = new_slide(prs, "实施方案", 7)
    add_picture_fit(s, img("image3.png"), 70, 105, 805, 275, pad=8)
    add_tag(s, "采集接入", 98, 410, 120, 42)
    add_tag(s, "缓存对齐", 252, 410, 120, 42)
    add_tag(s, "TDCA识别", 406, 410, 120, 42)
    add_tag(s, "门控决策", 560, 410, 120, 42, (255, 160, 70))
    add_tag(s, "设备执行", 714, 410, 120, 42)

    # 8 dynamic visual stimulation
    s = new_slide(prs, "动态刺激范式", 8)
    add_picture_fit(s, img("image5.jpeg"), 68, 100, 520, 335, pad=6)
    add_text(s, "动态摄像头范式", 625, 115, 250, 32, 25, (0, 246, 255), True)
    add_bullets(s, [
        "YOLO 轻量模型检测候选目标",
        "DeepSORT / ID 关联保持目标身份",
        "在目标边界叠加频率编码闪烁",
        "频率-标签-动作一一映射",
        "目标丢失时执行重分配策略",
    ], 620, 170, 265, size=17, gap=39)
    add_text(s, "用户注视真实目标，系统输出对应控制指令。", 100, 455, 760, 30, 22, (255, 255, 255), True, 2)

    # 9 data processing
    s = new_slide(prs, "数据处理", 9)
    add_picture_fit(s, img("image6.jpeg"), 66, 96, 365, 382, pad=6)
    add_text(s, "处理流水线", 475, 108, 220, 32, 25, (0, 246, 255), True)
    add_bullets(s, [
        "事件对齐与有效窗截取",
        "去直流、标准化、幅值约束",
        "滤波器组分解多频段信息",
        "时间延迟扩展增强短窗特征",
        "模板相关评分与子带加权融合",
    ], 475, 166, 360, size=18, gap=45)
    add_text(s, "原则：拒绝脏样本，保证训练与在线输入一致。", 475, 420, 365, 30, 21, (255, 80, 80), True, 2)

    # 10 TDCA and online decision
    s = new_slide(prs, "模型与决策", 10)
    add_picture_fit(s, img("image7.jpeg"), 72, 112, 405, 185, pad=5)
    add_picture_fit(s, img("image9.jpeg"), 72, 320, 405, 145, pad=5)
    add_text(s, "TDCA 负责“分得开”", 525, 120, 300, 30, 24, (0, 246, 255), True)
    add_text(s, "通过判别投影最大化类间差异、压缩类内波动，在短窗条件下提升类别可分性。", 525, 160, 315, 64, 17)
    add_text(s, "门控策略负责“不误触”", 525, 270, 300, 30, 24, (0, 246, 255), True)
    add_text(s, "用 Top1-Top2 分数差计算置信度，再叠加连续投票与冷却窗口，降低单次噪声带来的错误执行。", 525, 310, 315, 78, 17)
    add_tag(s, "准确率", 530, 430, 90, 36)
    add_tag(s, "误触发率", 645, 430, 105, 36, (255, 160, 70))
    add_tag(s, "响应时延", 775, 430, 105, 36)

    # 11 basis and conditions
    s = new_slide(prs, "基础条件", 11)
    add_picture_fit(s, img("image13.png"), 58, 105, 430, 288, pad=0)
    add_picture_fit(s, img("image11.png"), 525, 105, 290, 128, pad=4)
    add_picture_fit(s, img("image12.png"), 525, 255, 290, 128, pad=4)
    add_card(s, 72, 418, 225, 66, "技术基础", "已具备实时 EEG 接入、TDCA/CCA/FBCCA 对照链路。", body_size=14)
    add_card(s, 330, 418, 225, 66, "平台基础", "ROSMaster 小车、神舞 EEG 设备与刺激终端已就绪。", body_size=14)
    add_card(s, 588, 418, 225, 66, "指导基础", "导师团队具备脑机接口、模式识别与混合智能研究积累。", body_size=14)

    # 12 plan and outcomes
    s = new_slide(prs, "计划成果", 12)
    add_text(s, "实施计划", 72, 112, 150, 30, 26, (0, 246, 255), True)
    plan = [
        ("2026.04-06", "范式设计\n采集联调"),
        ("2026.07-10", "算法优化\n参数消融"),
        ("2026.11-2027.03", "动态场景\n闭环实验"),
        ("2027.04-06", "成果整理\n结题答辩"),
    ]
    for i, (t, b) in enumerate(plan):
        x = 90 + i * 205
        add_card(s, x, 165, 160, 116, t, b, body_size=16)
        if i < 3:
            add_arrow(s, x + 164, 223, x + 200, 223)
    add_text(s, "预期成果", 72, 332, 150, 30, 26, (0, 246, 255), True)
    add_tag(s, "创新训练报告", 95, 385, 160, 44)
    add_tag(s, "软件作品", 300, 385, 130, 44)
    add_tag(s, "实验数据与日志", 475, 385, 170, 44)
    add_tag(s, "竞赛 / 论文成果", 690, 385, 170, 44)
    add_text(s, "汇报完毕  恳请指导", 250, 470, 460, 42, 34, (255, 255, 255), True, 2)

    prs.SaveAs(OUT)
    prs.Close()
    try:
        app.Quit()
    except Exception:
        pass
    print(OUT)


if __name__ == "__main__":
    build()
