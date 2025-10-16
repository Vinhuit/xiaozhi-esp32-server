import requests
from bs4 import BeautifulSoup
from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from core.utils.util import get_ip_info

TAG = __name__
logger = setup_logging()

GET_WEATHER_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Tra cứu thời tiết tại một địa điểm. Người dùng nên nhập tên tỉnh/thành hoặc địa danh, ví dụ: Hà Nội, Đà Nẵng, Hải Phòng,... "
            "Nếu chỉ nhập tên tỉnh thì sẽ lấy thời tiết của thành phố trực thuộc tỉnh đó. Nếu người dùng hỏi 'thời tiết thế nào', không chỉ rõ nơi, thì location để trống."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Tên địa điểm, ví dụ: Hà Nội, Huế, Vũng Tàu... (Không bắt buộc, không nhập thì sẽ tự xác định theo IP hoặc mặc định Hà Nội)",
                },
                "lang": {
                    "type": "string",
                    "description": "Mã ngôn ngữ trả về, ví dụ: vi_VN, en_US,... Mặc định là vi_VN.",
                },
            },
            "required": ["lang"],
        },
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    )
}

# Bảng mã thời tiết => mô tả tiếng Việt
WEATHER_CODE_MAP = {
    "100": "Trời quang",
    "101": "Nhiều mây",
    "102": "Ít mây",
    "103": "Trời quang từng lúc có mây",
    "104": "Âm u",
    "150": "Trời quang",
    "151": "Nhiều mây",
    "152": "Ít mây",
    "153": "Trời quang từng lúc có mây",
    "300": "Mưa rào",
    "301": "Mưa rào mạnh",
    "302": "Dông rải rác",
    "303": "Dông mạnh",
    "304": "Dông kèm mưa đá",
    "305": "Mưa nhỏ",
    "306": "Mưa vừa",
    "307": "Mưa to",
    "308": "Mưa cực lớn",
    "309": "Mưa phùn",
    "310": "Mưa rất to",
    "311": "Mưa đặc biệt lớn",
    "312": "Mưa cực kỳ lớn",
    "313": "Mưa đóng băng",
    "314": "Mưa nhỏ đến vừa",
    "315": "Mưa vừa đến to",
    "316": "Mưa to đến rất to",
    "317": "Mưa rất to đến đặc biệt lớn",
    "318": "Mưa đặc biệt lớn đến cực kỳ lớn",
    "350": "Mưa rào",
    "351": "Mưa rào mạnh",
    "399": "Có mưa",
    "400": "Tuyết nhẹ",
    "401": "Tuyết vừa",
    "402": "Tuyết dày",
    "403": "Bão tuyết",
    "404": "Mưa tuyết",
    "405": "Thời tiết mưa tuyết",
    "406": "Mưa rào tuyết",
    "407": "Tuyết rào",
    "408": "Tuyết nhẹ đến vừa",
    "409": "Tuyết vừa đến dày",
    "410": "Tuyết dày đến rất dày",
    "456": "Mưa rào tuyết",
    "457": "Tuyết rào",
    "499": "Có tuyết",
    "500": "Sương nhẹ",
    "501": "Sương mù",
    "502": "Sương mù bụi",
    "503": "Cát bay",
    "504": "Bụi mù",
    "507": "Bão cát",
    "508": "Bão cát mạnh",
    "509": "Sương mù dày",
    "510": "Sương mù cực dày",
    "511": "Sương mù trung bình",
    "512": "Sương mù nặng",
    "513": "Sương mù nghiêm trọng",
    "514": "Sương mù lớn",
    "515": "Sương mù cực kỳ lớn",
    "900": "Nóng",
    "901": "Lạnh",
    "999": "Không xác định",
}

def fetch_city_info(location, api_key, api_host):
    url = f"https://{api_host}/geo/v2/city/lookup?key={api_key}&location={location}&lang=vi"
    response = requests.get(url, headers=HEADERS).json()
    return response.get("location", [])[0] if response.get("location") else None

def fetch_weather_page(url):
    response = requests.get(url, headers=HEADERS)
    return BeautifulSoup(response.text, "html.parser") if response.ok else None

def parse_weather_info(soup):
    city_name = soup.select_one("h1.c-submenu__location").get_text(strip=True)

    current_abstract = soup.select_one(".c-city-weather-current .current-abstract")
    current_abstract = (
        current_abstract.get_text(strip=True) if current_abstract else "Không rõ"
    )

    current_basic = {}
    for item in soup.select(
        ".c-city-weather-current .current-basic .current-basic___item"
    ):
        parts = item.get_text(strip=True, separator=" ").split(" ")
        if len(parts) == 2:
            key, value = parts[1], parts[0]
            current_basic[key] = value

    temps_list = []
    for row in soup.select(".city-forecast-tabs__row")[:7]:  # Lấy 7 ngày dự báo
        date = row.select_one(".date-bg .date").get_text(strip=True)
        weather_code = (
            row.select_one(".date-bg .icon")["src"].split("/")[-1].split(".")[0]
        )
        weather = WEATHER_CODE_MAP.get(weather_code, "Không xác định")
        temps = [span.get_text(strip=True) for span in row.select(".tmp-cont .temp")]
        high_temp, low_temp = (temps[0], temps[-1]) if len(temps) >= 2 else (None, None)
        temps_list.append((date, weather, high_temp, low_temp))

    return city_name, current_abstract, current_basic, temps_list

@register_function("get_weather", GET_WEATHER_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_weather(conn, location: str = None, lang: str = "vi_VN"):
    from core.utils.cache.manager import cache_manager, CacheType

    api_host = conn.config["plugins"]["get_weather"].get(
        "api_host", "mj7p3y7naa.re.qweatherapi.com"
    )
    api_key = conn.config["plugins"]["get_weather"].get(
        "api_key", "a861d0d5e7bf4ee1a83d9a9e4f96d4da"
    )
    default_location = conn.config["plugins"]["get_weather"]["default_location"]
    client_ip = conn.client_ip

    # Ưu tiên lấy địa điểm do người dùng nhập
    if not location:
        # Nếu không nhập, xác định theo IP hoặc mặc định
        if client_ip:
            cached_ip_info = cache_manager.get(CacheType.IP_INFO, client_ip)
            if cached_ip_info:
                location = cached_ip_info.get("city")
            else:
                ip_info = get_ip_info(client_ip, logger)
                if ip_info:
                    cache_manager.set(CacheType.IP_INFO, client_ip, ip_info)
                    location = ip_info.get("city")

            if not location:
                location = default_location
        else:
            location = default_location

    # Thử lấy cache
    weather_cache_key = f"full_weather_{location}_{lang}"
    cached_weather_report = cache_manager.get(CacheType.WEATHER, weather_cache_key)
    if cached_weather_report:
        return ActionResponse(Action.REQLLM, cached_weather_report, None)

    # Nếu không có cache, lấy dữ liệu trực tiếp
    city_info = fetch_city_info(location, api_key, api_host)
    if not city_info:
        return ActionResponse(
            Action.REQLLM, f"Không tìm thấy địa điểm: {location}. Vui lòng kiểm tra lại tên địa phương.", None
        )
    soup = fetch_weather_page(city_info["fxLink"])
    if not soup:
        return ActionResponse(Action.REQLLM, None, "Lấy dữ liệu thời tiết thất bại!")
    city_name, current_abstract, current_basic, temps_list = parse_weather_info(soup)

    weather_report = f"Bạn đang tra cứu thời tiết tại: {city_name}\n\nThời tiết hiện tại: {current_abstract}\n"

    if current_basic:
        weather_report += "Các thông số khác:\n"
        for key, value in current_basic.items():
            if value != "0":
                weather_report += f"  · {key}: {value}\n"

    weather_report += "\nDự báo 7 ngày tới:\n"
    for date, weather, high, low in temps_list:
        weather_report += f"{date}: {weather}, nhiệt độ {low}~{high}\n"

    weather_report += "\n(Nếu bạn muốn tra cứu một ngày cụ thể, hãy nói rõ ngày đó nhé!)"

    cache_manager.set(CacheType.WEATHER, weather_cache_key, weather_report)

    return ActionResponse(Action.REQLLM, weather_report, None)
