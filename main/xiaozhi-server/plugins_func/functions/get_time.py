from datetime import datetime
import cnlunar
from plugins_func.register import register_function, ToolType, ActionResponse, Action

get_lunar_function_desc = {
    "type": "function",
    "function": {
        "name": "get_lunar",
        "description": (
            "Tra cứu thông tin lịch âm, lịch vạn niên, ngày hoàng đạo cho một ngày cụ thể. "
            "Bạn có thể hỏi về ngày âm, can chi, tiết khí, con giáp, cung hoàng đạo, bát tự, các việc nên làm - kiêng kỵ,... "
            "Nếu không chỉ rõ nội dung cần hỏi, mặc định trả về năm can chi và ngày âm. "
            "Các câu hỏi kiểu 'Hôm nay là ngày âm gì?', 'Âm lịch hôm nay' nên trả lời trực tiếp bằng context, không gọi lại hàm này."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Ngày cần tra cứu, định dạng YYYY-MM-DD, ví dụ 2024-01-01. Nếu không nhập sẽ lấy ngày hiện tại.",
                },
                "query": {
                    "type": "string",
                    "description": "Nội dung cần hỏi, ví dụ ngày âm, can chi, lễ hội, tiết khí, con giáp, cung hoàng đạo, bát tự, việc nên - kỵ,...",
                },
            },
            "required": [],
        },
    },
}

@register_function("get_lunar", get_lunar_function_desc, ToolType.WAIT)
def get_lunar(date=None, query=None):
    """
    Trả về thông tin lịch âm, hoàng đạo, can chi, tiết khí, con giáp, cung hoàng đạo, bát tự, việc nên - kỵ... cho một ngày bất kỳ.
    """
    from core.utils.cache.manager import cache_manager, CacheType

    # Dùng ngày nhập vào, hoặc lấy ngày hiện tại
    if date:
        try:
            now = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return ActionResponse(
                Action.REQLLM,
                "Định dạng ngày không hợp lệ. Hãy nhập đúng định dạng YYYY-MM-DD, ví dụ: 2024-01-01",
                None,
            )
    else:
        now = datetime.now()

    current_date = now.strftime("%Y-%m-%d")

    if query is None:
        query = "Mặc định: năm can chi và ngày âm lịch"

    # Kiểm tra cache
    lunar_cache_key = f"lunar_info_{current_date}"
    cached_lunar_info = cache_manager.get(CacheType.LUNAR, lunar_cache_key)
    if cached_lunar_info:
        return ActionResponse(Action.REQLLM, cached_lunar_info, None)

    response_text = (
        f"Thông tin lịch âm và hoàng đạo cho ngày {current_date} (nội dung bạn hỏi: {query}):\n"
    )

    lunar = cnlunar.Lunar(now, godType="8char")
    response_text += (
        f"- Ngày âm lịch: {lunar.lunarYearCn} năm, tháng {lunar.lunarMonthCn[:-1]}, ngày {lunar.lunarDayCn}\n"
        f"- Can chi: {lunar.year8Char} năm, {lunar.month8Char} tháng, {lunar.day8Char} ngày\n"
        f"- Con giáp: {lunar.chineseYearZodiac}\n"
        f"- Bát tự: {' '.join([lunar.year8Char, lunar.month8Char, lunar.day8Char, lunar.twohour8Char])}\n"
        f"- Lễ/tết hôm nay: {', '.join(filter(None, (lunar.get_legalHolidays(), lunar.get_otherHolidays(), lunar.get_otherLunarHolidays())))}\n"
        f"- Tiết khí hôm nay: {lunar.todaySolarTerms}\n"
        f"- Tiết khí tiếp theo: {lunar.nextSolarTerm} (ngày {lunar.nextSolarTermYear}-{lunar.nextSolarTermDate[0]}-{lunar.nextSolarTermDate[1]})\n"
        f"- Danh sách tiết khí trong năm: {', '.join([f'{term}({date[0]}/{date[1]})' for term, date in lunar.thisYearSolarTermsDic.items()])}\n"
        f"- Xung khắc (tránh): {lunar.chineseZodiacClash}\n"
        f"- Cung hoàng đạo: {lunar.starZodiac}\n"
        f"- Nạp âm: {lunar.get_nayin()}\n"
        f"- Tránh kỵ (Bách kỵ): {lunar.get_pengTaboo(delimit=', ')}\n"
        f"- Trực ngày: {lunar.get_today12DayOfficer()[0]}\n"
        f"- Thần trực: {lunar.get_today12DayOfficer()[1]} ({lunar.get_today12DayOfficer()[2]})\n"
        f"- Nhị thập bát tú: {lunar.get_the28Stars()}\n"
        f"- Hướng thần tài, hỷ thần: {' '.join(lunar.get_luckyGodsDirection())}\n"
        f"- Thai thần: {lunar.get_fetalGod()}\n"
        f"- Việc nên làm (10 mục đầu): {'; '.join(lunar.goodThing[:10])}\n"
        f"- Việc kiêng kỵ (10 mục đầu): {'; '.join(lunar.badThing[:10])}\n"
        "(Nếu không hỏi cụ thể về việc nên/kỵ thì mặc định chỉ trả về năm can chi và ngày âm lịch)"
    )

    # Cache lại kết quả
    cache_manager.set(CacheType.LUNAR, lunar_cache_key, response_text)

    return ActionResponse(Action.REQLLM, response_text, None)
