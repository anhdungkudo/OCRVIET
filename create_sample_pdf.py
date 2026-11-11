"""Tạo file PDF mẫu mô phỏng văn bản hành chính Việt Nam có dấu mộc đỏ đè lên chữ."""

from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz


def generate_sample_pdf(output_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 standard at 72 DPI

    # 1. Header Table
    # Cột trái
    page.insert_text((50, 60), "ỦY BAN NHÂN DÂN", fontsize=11, fontname="times-bold")
    page.insert_text((40, 75), "THÀNH PHỐ HÀ NỘI", fontsize=11, fontname="times-bold")
    page.insert_text((35, 90), "SỞ THÔNG TIN VÀ TRUYỀN THÔNG", fontsize=10, fontname="times-bold")
    page.insert_text((65, 110), "Số: 125/QĐ-STTTT", fontsize=11, fontname="times-roman")

    # Cột phải
    page.insert_text((310, 60), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", fontsize=11, fontname="times-bold")
    page.insert_text((350, 75), "Độc lập - Tự do - Hạnh phúc", fontsize=11, fontname="times-bold")
    # Đường kẻ dưới tiêu ngữ
    page.draw_line((365, 82), (475, 82), color=(0, 0, 0), width=1)
    page.insert_text((330, 110), "Hà Nội, ngày 10 tháng 11 năm 2026", fontsize=11, fontname="times-italic")

    # 2. Title Block
    page.insert_text((245, 155), "QUYẾT ĐỊNH", fontsize=14, fontname="times-bold")
    page.insert_text(
        (135, 175),
        "Về việc ban hành quy chế tiếp nhận và số hóa hồ sơ thủ tục hành chính",
        fontsize=12,
        fontname="times-bold",
    )

    # 3. Body Blocks
    page.insert_text(
        (85, 210),
        "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015;",
        fontsize=12,
        fontname="times-italic",
    )
    page.insert_text(
        (85, 230),
        "Căn cứ Nghị định số 30/2020/NĐ-CP ngày 05 tháng 3 năm 2020 của Chính phủ về công tác văn thư;",
        fontsize=12,
        fontname="times-italic",
    )
    page.insert_text(
        (85, 250),
        "Theo đề nghị của Chánh Văn phòng Sở Thông tin và Truyền thông.",
        fontsize=12,
        fontname="times-italic",
    )

    page.insert_text((250, 280), "QUYẾT ĐỊNH:", fontsize=12, fontname="times-bold")

    page.insert_text(
        (85, 305),
        "Điều 1. Ban hành kèm theo Quyết định này Quy chế tiếp nhận và số hóa hồ sơ thủ tục hành chính.",
        fontsize=12,
        fontname="times-roman",
    )
    page.insert_text(
        (85, 330),
        "Điều 2. Bảng phân công chỉ tiêu số hóa các phòng ban chuyên môn:",
        fontsize=12,
        fontname="times-roman",
    )

    # Bảng số liệu
    table_rect = fitz.Rect(50, 345, 545, 435)
    page.draw_rect(table_rect, color=(0, 0, 0), width=1)
    # Hàng ngang
    page.draw_line((50, 370), (545, 370), color=(0, 0, 0), width=1)
    page.draw_line((50, 395), (545, 395), color=(0, 0, 0), width=1)
    page.draw_line((50, 415), (545, 415), color=(0, 0, 0), width=1)
    # Cột dọc
    page.draw_line((90, 345), (90, 435), color=(0, 0, 0), width=1)
    page.draw_line((290, 345), (290, 435), color=(0, 0, 0), width=1)
    page.draw_line((420, 345), (420, 435), color=(0, 0, 0), width=1)

    # Tiêu đề bảng
    page.insert_text((60, 362), "STT", fontsize=10, fontname="times-bold")
    page.insert_text((150, 362), "Tên đơn vị", fontsize=10, fontname="times-bold")
    page.insert_text((330, 362), "Chỉ tiêu số hóa", fontsize=10, fontname="times-bold")
    page.insert_text((455, 362), "Tiến độ hoàn thành", fontsize=10, fontname="times-bold")

    # Hàng 1
    page.insert_text((65, 386), "1", fontsize=10, fontname="times-roman")
    page.insert_text((100, 386), "Văn phòng Sở", fontsize=10, fontname="times-roman")
    page.insert_text((345, 386), "100%", fontsize=10, fontname="times-roman")
    page.insert_text((460, 386), "Tháng 12/2026", fontsize=10, fontname="times-roman")

    # Hàng 2
    page.insert_text((65, 408), "2", fontsize=10, fontname="times-roman")
    page.insert_text((100, 408), "Phòng CNTT", fontsize=10, fontname="times-roman")
    page.insert_text((345, 408), "100%", fontsize=10, fontname="times-roman")
    page.insert_text((460, 408), "Tháng 12/2026", fontsize=10, fontname="times-roman")

    # Hàng 3
    page.insert_text((65, 428), "3", fontsize=10, fontname="times-roman")
    page.insert_text((100, 428), "Phòng Bưu chính - Viễn thông", fontsize=10, fontname="times-roman")
    page.insert_text((345, 428), "95%", fontsize=10, fontname="times-roman")
    page.insert_text((460, 428), "Tháng 12/2026", fontsize=10, fontname="times-roman")

    page.insert_text(
        (85, 460),
        "Điều 3. Chánh Văn phòng, Trưởng các phòng chuyên môn chịu trách nhiệm thi hành Quyết định này./.",
        fontsize=12,
        fontname="times-roman",
    )

    # 4. Footer Table
    # Cột trái Nơi nhận
    page.insert_text((50, 520), "Nơi nhận:", fontsize=11, fontname="times-bolditalic")
    page.insert_text((50, 535), "- Như Điều 3;", fontsize=10, fontname="times-roman")
    page.insert_text((50, 550), "- UBND Thành phố (để b/c);", fontsize=10, fontname="times-roman")
    page.insert_text((50, 565), "- Giám đốc, các PGĐ Sở;", fontsize=10, fontname="times-roman")
    page.insert_text((50, 580), "- Lưu: VT, CTTĐT (3b).", fontsize=10, fontname="times-roman")

    # Cột phải Người ký
    page.insert_text((380, 520), "KT. GIÁM ĐỐC", fontsize=12, fontname="times-bold")
    page.insert_text((375, 538), "PHÓ GIÁM ĐỐC", fontsize=12, fontname="times-bold")
    page.insert_text((380, 640), "Nguyễn Văn Tuấn", fontsize=12, fontname="times-bold")

    # Chữ ký xanh mô phỏng
    page.draw_line((385, 570), (430, 595), color=(0.1, 0.2, 0.8), width=2)
    page.draw_line((430, 595), (460, 560), color=(0.1, 0.2, 0.8), width=2)
    page.draw_line((410, 585), (475, 600), color=(0.1, 0.2, 0.8), width=2)

    # Con dấu đỏ tròn đè lên 1/3 chữ ký và đè lên chức vụ / họ tên
    shape = page.new_shape()
    seal_center = (380, 580)
    # Vòng tròn ngoài
    shape.draw_circle(seal_center, 45)
    # Vòng tròn trong
    shape.draw_circle(seal_center, 30)
    shape.finish(
        color=(0.85, 0.1, 0.1),
        fill=(0.9, 0.15, 0.15),
        stroke_opacity=0.85,
        fill_opacity=0.18,
        width=2.5,
    )
    shape.commit()

    page.insert_text(
        (345, 575), "SỞ THÔNG TIN", fontsize=9, color=(0.85, 0.1, 0.1), fontname="times-bold"
    )
    page.insert_text(
        (348, 590), "VÀ TRUYỀN THÔNG", fontsize=8, color=(0.85, 0.1, 0.1), fontname="times-bold"
    )

    doc.save(str(output_path))
    doc.close()
    print(f"Đã tạo file PDF mẫu thành công tại: {output_path}")


if __name__ == "__main__":
    generate_sample_pdf(Path("sample_van_ban_scan.pdf"))
