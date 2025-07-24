from plugins_func.register import register_function, ToolType, ActionResponse, Action
from plugins_func.functions.hass_init import initialize_hass_handler
from config.logger import setup_logging
import asyncio
import requests

TAG = __name__
logger = setup_logging()

hass_get_state_function_desc = {
    "type": "function",
    "function": {
        "name": "hass_get_state",
        "description": "获取homeassistant里设备的状态,包括查询灯光亮度、颜色、色温,媒体播放器的音量,设备的暂停、继续操作",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "需要操作的设备id,homeassistant里的entity_id",
                }
            },
            "required": ["entity_id"],
        },
    },
}


@register_function("hass_get_state", hass_get_state_function_desc, ToolType.SYSTEM_CTL)
def hass_get_state(conn, entity_id=""):
    try:

        future = asyncio.run_coroutine_threadsafe(
            handle_hass_get_state(conn, entity_id), conn.loop
        )
        # 添加10秒超时
        ha_response = future.result(timeout=10)
        return ActionResponse(Action.REQLLM, ha_response, None)
    except asyncio.TimeoutError:
        logger.bind(tag=TAG).error("获取Home Assistant状态超时")
        return ActionResponse(Action.ERROR, "请求超时", None)
    except Exception as e:
        error_msg = f"执行Home Assistant操作失败"
        logger.bind(tag=TAG).error(error_msg)
        return ActionResponse(Action.ERROR, error_msg, None)



async def handle_hass_get_state(conn, entity_id):
    ha_config = initialize_hass_handler(conn)
    api_key = ha_config.get("api_key")
    base_url = ha_config.get("base_url")

    if not entity_id:
        logger.bind(tag=TAG).error("Invalid device ID: {}".format(entity_id))
        return "Invalid device ID"

    url = f"{base_url}/api/states/{entity_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


    logger.bind(tag=TAG).info(f"Querying device state: {url} with headers: {headers}")
    try:
        response = await asyncio.to_thread(requests.get, url, headers=headers)
    except Exception as e:
        logger.bind(tag=TAG).error(f"Network request error: {e}")
        return "Network request failed"

    if response.status_code == 200:
        try:
            json_data = response.json()
            attributes = json_data.get("attributes", {})
            state = json_data.get("state", "unknown")
            responsetext = f"Device status: {state}"

            for key, value in attributes.items():
                responsetext += f"\n{key}: {value}"
            if "media_title" in response.json()["attributes"]:
                responsetext = (
                    responsetext
                    + "正在播放的是:"
                    + str(response.json()["attributes"]["media_title"])
                    + " "
                )
            if "volume_level" in response.json()["attributes"]:
                responsetext = (
                    responsetext
                    + "音量是:"
                    + str(response.json()["attributes"]["volume_level"])
                    + " "
                )
            if "color_temp_kelvin" in response.json()["attributes"]:
                responsetext = (
                    responsetext
                    + "色温是:"
                    + str(response.json()["attributes"]["color_temp_kelvin"])
                    + " "
                )
            if "rgb_color" in response.json()["attributes"]:
                responsetext = (
                    responsetext
                    + "rgb颜色是:"
                    + str(response.json()["attributes"]["rgb_color"])
                    + " "
                )
            if "brightness" in response.json()["attributes"]:
                responsetext = (
                    responsetext
                    + "亮度是:"
                    + str(response.json()["attributes"]["brightness"])
                    + " "
                )

            logger.bind(tag=TAG).info(f"Query return content: {responsetext}")
            return responsetext
        except (KeyError, ValueError) as e:
            logger.bind(tag=TAG).error(f"Parse JSON data error: {e}")
            return "Parse JSON data failed"
    elif response.status_code == 404:
        return f"Device '{entity_id}' not found"
    else:
        return f"Failed to query device, error code: {response.status_code}"
