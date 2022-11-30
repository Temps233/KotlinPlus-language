_m = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}

release_year  = 2022
release_month = 11
release_day   = 28

release = f"{_m.get(release_month)} {release_day} {release_year}"
release_info = f"0.1.0 (alpha, {release})"