import os
import win32com.client as win32


OUT = r"C:\Users\adam\Desktop\基于脑机接口与实时视觉反馈的交互系统_立项报告_图文版.pptx"
IMG_DIR = os.path.abspath("申报书_images")


def rgb(r, g, b):
    return int(r) + int(g) * 256 + int(b) * 65536


def add_text(slide, text, x, y, w, h, size=24, color=(255, 255, 255),
             bold=False, font="Microsoft YaHei", align=1):
    box = slide.Shapes.AddTextbox(1, x, y, w, h)
    tr = box.TextFrame.TextRange
    tr.Text = text
    tr.Font.Name = font
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = rgb(*color)
    tr.ParagraphFormat.Alignment = align
    box.TextFrame.MarginLeft = 4
    box.TextFrame.MarginRight = 4
    box.TextFrame.MarginTop = 2
    box.TextFrame.MarginBottom = 2
    return box


def add_card(slide, x, y, w, h, title, body, accent=(0, 240, 255), body_size=20):
    shape = slide.Shapes.AddShape(5, x, y, w, h)
    shape.Fill.ForeColor.RGB = rgb(13, 45, 82)
    shape.Fill.Transparency = 0.08
    shape.Line.ForeColor.RGB = rgb(*accent)
    shape.Line.Transparency = 0.2
    shape.Line.Weight = 1.4
    add_text(slide, title, x + 16, y + 14, w - 32, 34, 23, accent, True)
    add_text(slide, body, x + 18, y + 54, w - 36, h - 62, body_size, (238, 246, 255))
    return shape


def img(name):
    return os.path.join(IMG_DIR, name)


def add_picture_fit(slide, path, x, y, w, h, pad=8, border=True):
    panel = slide.Shapes.AddShape(1, x, y, w, h)
    panel.Fill.ForeColor.RGB = rgb(246, 250, 255)
    panel.Fill.Transparency = 0
    panel.Line.ForeColor.RGB = rgb(0, 210, 240)
    panel.Line.Transparency = 0.15
    panel.Line.Weight = 1.2

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
    if not border:
        panel.Line.Visible = 0
    return pic


def add_header(slide, section):
    add_text(slide, "项目内容", 48, 28, 165, 46, 30, (255, 255, 255), True)
    line = slide.Shapes.AddLine(220, 54, 295, 54)
    line.Line.ForeColor.RGB = rgb(255, 255, 255)
    line.Line.Weight = 2.2
    add_text(slide, section, 306, 32, 360, 42, 24, (255, 255, 255), True)
    # Corner badges, replacing hard dependency on exact logo assets.
    # Some Office COM sessions reject editing oval line properties, so use text badges.
    add_text(slide, "WYU · BCI", 790, 30, 120, 28, 14, (235, 250, 255), True, align=2)


def add_bg(slide, prs):
    slide.FollowMasterBackground = False
    slide.Background.Fill.ForeColor.RGB = rgb(3, 21, 55)
    # bottom wave bands
    for i, col in enumerate([(7, 49, 93), (11, 78, 125), (0, 158, 205)]):
        band = slide.Shapes.AddShape(1, -40, 405 + i * 32, prs.PageSetup.SlideWidth + 80, 120)
        band.Fill.ForeColor.RGB = rgb(*col)
        band.Fill.Transparency = 0.50 + i * 0.10
        band.Line.Visible = 0
    # subtle grid points
    for x in range(80, 900, 110):
        for y in range(130, 395, 70):
            dot = slide.Shapes.AddShape(9, x, y, 3, 3)
            dot.Fill.ForeColor.RGB = rgb(0, 238, 255)
            dot.Fill.Transparency = 0.35
            dot.Line.Visible = 0
    # left light rail
    rail = slide.Shapes.AddLine(36, 32, 36, 500)
    rail.Line.ForeColor.RGB = rgb(0, 240, 255)
    rail.Line.Transparency = 0.35
    rail.Line.Weight = 1.2


def new_slide(prs, section=None):
    slide = prs.Slides.Add(prs.Slides.Count + 1, 12)
    add_bg(slide, prs)
    if section:
        add_header(slide, section)
    return slide


def add_flow(slide, items, x, y, w, h):
    gap = 16
    box_w = (w - gap * (len(items) - 1)) / len(items)
    for i, (title, desc) in enumerate(items):
        bx = x + i * (box_w + gap)
        add_card(slide, bx, y, box_w, h, title, desc, body_size=16)
        if i < len(items) - 1:
            line = slide.Shapes.AddLine(bx + box_w + 2, y + h / 2, bx + box_w + gap - 2, y + h / 2)
            line.Line.ForeColor.RGB = rgb(0, 240, 255)
            line.Line.Weight = 2


def build():
    app = win32.Dispatch("KWPP.Application")
    try:
        app.Visible = True
    except Exception:
        pass
    if os.path.exists(OUT):
        os.remove(OUT)
    prs = app.Presentations.Add()
    prs.PageSetup.SlideWidth = 960
    prs.PageSetup.SlideHeight = 540

    # 1 cover
    s = new_slide(prs)
    add_text(s, "基于脑机接口与实时视觉反馈的交互系统", 120, 150, 720, 95, 42, (0, 246, 255), True, align=2)
    add_text(s, "大学生创新训练计划项目立项报告", 210, 260, 540, 44, 25, (245, 250, 255), True, align=2)
    add_text(s, "负责人：张连成    指导教师：王洪涛教授\n电子与信息工程学院 · 计算机科学与技术", 260, 332, 440, 55, 17, (230, 242, 255), align=2)
    add_text(s, "2026", 790, 430, 90, 36, 22, (0, 240, 255), True, align=2)

    # 2 overview
    s = new_slide(prs, "项目概况")
    add_picture_fit(s, img("image1.png"), 52, 105, 555, 330, pad=0)
    add_card(s, 635, 112, 260, 92, "项目定位", "面向真实交互场景的非侵入式 SSVEP-BCI 系统", body_size=18)
    add_card(s, 635, 222, 260, 92, "核心范式", "实时视觉反馈 + 动态闪烁刺激 + TDCA 在线解码", body_size=18)
    add_card(s, 635, 332, 260, 92, "实施周期", "2026.04 - 2027.06\n国家级创新训练项目申报", body_size=18)

    # 3 background
    s = new_slide(prs, "项目背景")
    add_picture_fit(s, img("image5.jpeg"), 58, 106, 560, 348, pad=6)
    add_text(s, "创新点：把刺激界面叠加到真实目标上", 650, 118, 250, 70, 27, (0, 246, 255), True)
    add_text(s, "传统 BCI 多为抽象矩阵按钮，用户需要先理解映射关系。\n\n本项目用摄像头画面承载刺激，用户注视真实物体即可产生对应 SSVEP 指令。\n\n目标：让交互从“看按钮”走向“看目标”。",
             650, 205, 245, 230, 20, (245, 250, 255))

    # 4 purpose
    s = new_slide(prs, "目的与意义")
    add_picture_fit(s, img("image13.png"), 58, 110, 505, 335, pad=0)
    add_card(s, 600, 112, 285, 92, "实践意义", "助残康复：降低抽象指令学习成本，提升辅助设备可用性。", body_size=17)
    add_card(s, 600, 222, 285, 92, "工程意义", "构建“采集-解码-控制-反馈”可复盘闭环。", body_size=17)
    add_card(s, 600, 332, 285, 92, "拓展意义", "可迁移到小车、护理设备、高危远程操作等场景。", body_size=17)

    # 5 key problems
    s = new_slide(prs, "关键问题")
    add_picture_fit(s, img("image4.jpeg"), 60, 112, 570, 315, pad=6)
    add_text(s, "要解决的不只是分类准确率", 665, 120, 230, 58, 26, (0, 246, 255), True)
    add_text(s, "1. 刺激、采集、分析窗口的时序对齐\n2. 短窗条件下的噪声放大\n3. 离线训练与在线测试口径一致\n4. 判决输出到设备动作的闭环延迟",
             665, 205, 225, 175, 20, (245, 250, 255))
    add_text(s, "关键抓手：统一时间戳与有效分析窗", 650, 395, 260, 34, 21, (255, 70, 70), True, align=2)

    # 6 research content
    s = new_slide(prs, "研究内容")
    add_picture_fit(s, img("image3.png"), 60, 105, 820, 275, pad=6)
    add_card(s, 78, 405, 245, 65, "链路一体化", "采集、缓存、预处理、识别、决策、执行统一建模", body_size=15)
    add_card(s, 358, 405, 245, 65, "可回溯", "状态反馈与日志记录贯穿全流程", body_size=15)
    add_card(s, 638, 405, 245, 65, "可部署", "执行端面向 ROSMaster1 控制", body_size=15)

    # 7 route
    s = new_slide(prs, "技术路线")
    add_picture_fit(s, img("image10.jpeg"), 70, 103, 500, 350, pad=5)
    add_text(s, "五层软件架构", 620, 118, 230, 38, 28, (0, 246, 255), True)
    add_text(s, "L1 应用展示层\nL2 控制决策层\nL3 算法解码层\nL4 信号处理层\nL5 设备接入层",
             620, 172, 240, 145, 22, (255, 255, 255))
    add_text(s, "优势：模块解耦、配置可管、日志可追、接口可扩展。", 620, 340, 250, 68, 20, (245, 250, 255))

    # 8 algorithm
    s = new_slide(prs, "核心算法")
    add_picture_fit(s, img("image6.jpeg"), 60, 96, 345, 390, pad=6)
    add_picture_fit(s, img("image7.jpeg"), 435, 116, 430, 185, pad=6)
    add_picture_fit(s, img("image8.jpeg"), 435, 320, 430, 150, pad=6)
    add_text(s, "TDCA 将时间延迟扩展、判别投影、模板相关和子带融合串成短窗解码链路。", 430, 86, 460, 28, 17, (0, 246, 255), True, align=2)

    # 9 basis and plan
    s = new_slide(prs, "在线决策")
    add_picture_fit(s, img("image9.jpeg"), 60, 100, 455, 365, pad=6)
    add_text(s, "门控策略", 570, 122, 230, 40, 30, (0, 246, 255), True)
    add_text(s, "输入 TDCA 分数向量后，先计算最高分与次高分的差值作为置信度。\n\n只有当置信度超过阈值，并且连续投票一致时，才下发控制指令。\n\n冷却窗口用于防止一次意图被重复执行。",
             570, 188, 285, 220, 20, (245, 250, 255))
    add_text(s, "目标：降低误触发，同时保留实时响应。", 568, 420, 300, 30, 21, (255, 70, 70), True)

    # 10 outcomes
    s = new_slide(prs, "平台与成果")
    add_picture_fit(s, img("image11.png"), 58, 102, 300, 230, pad=5)
    add_picture_fit(s, img("image12.png"), 382, 102, 300, 230, pad=5)
    add_card(s, 705, 105, 175, 90, "硬件平台", "ROSMaster 小车\n神舞 EEG 采集设备\n刺激与执行终端", body_size=15)
    add_card(s, 705, 215, 175, 90, "预期交付", "创新训练报告\n软件作品\n竞赛或论文成果", body_size=15)
    add_text(s, "汇报完毕  恳请指导", 190, 390, 580, 55, 42, (255, 255, 255), True, align=2)
    add_text(s, "基于脑机接口与实时视觉反馈的交互系统", 260, 455, 440, 28, 17, (0, 246, 255), align=2)

    prs.SaveAs(OUT)
    prs.Close()
    app.Quit()
    return OUT


if __name__ == "__main__":
    print(build())
