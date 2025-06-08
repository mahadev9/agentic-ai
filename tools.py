import os
import json
import math
from datetime import datetime, timezone

import pytz
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.tools import Tool, tool


google_search_api_wrapper = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
)

web_search_tool = Tool(
    name="web_search",
    description="Search the web for relevant documents",
    func=google_search_api_wrapper.run,
    k=25,
)

weather_wrapper = OpenWeatherMapAPIWrapper(
    openweathermap_api_key=os.getenv("OPEN_WEATHER_MAP_KEY"),
)

weather_tool = Tool(
    name="weather",
    description="Get current weather information for a specified location",
    func=weather_wrapper.run,
)


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "sqrt(16)", "sin(pi/2)")

    Returns:
        JSON string with calculation result
    """
    try:
        # Safe mathematical functions
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "ceil": math.ceil,
            "floor": math.floor,
            "factorial": math.factorial,
        }

        # Evaluate the expression safely
        result = eval(expression, safe_dict)

        return json.dumps(
            {"expression": expression, "result": result, "type": type(result).__name__},
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"error": "Calculation failed", "expression": expression, "message": str(e)}
        )


@tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """
    Get current date and time information.

    Args:
        timezone_name: Timezone name (e.g., "UTC", "US/Eastern", "Europe/London", "Asia/Tokyo")

    Returns:
        JSON string with time information
    """
    try:

        utc_now = datetime.now(timezone.utc)

        if timezone_name.upper() != "UTC":
            try:
                target_tz = pytz.timezone(timezone_name)
                local_time = utc_now.astimezone(target_tz)
            except pytz.exceptions.UnknownTimeZoneError:
                local_time = utc_now
                timezone_name = "UTC (requested timezone not found)"
        else:
            local_time = utc_now

        time_info = {
            "current_time": local_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone_name,
            "iso_format": local_time.isoformat(),
            "weekday": local_time.strftime("%A"),
            "day_of_year": local_time.timetuple().tm_yday,
            "week_number": local_time.isocalendar()[1],
            "unix_timestamp": int(local_time.timestamp()),
        }

        return json.dumps(time_info, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "error": "Time request failed",
                "message": str(e),
                "utc_fallback": datetime.now(timezone.utc).isoformat(),
            }
        )
