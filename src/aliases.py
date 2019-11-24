to_drop = [
    "Placement - Time",
    "Arrival at Destination - Time",
    "Confirmation - Time",
    "Arrival at Pickup - Time",
    "Pickup - Time",
    "Vehicle Type",
    "Temperature",
]

to_datetime = [
    "Placement - Time",
    "Confirmation - Time",
    "Arrival at Pickup - Time",
    "Pickup - Time",
]

not_to_encoded = [
    "Distance (KM)",
    "Pickup Lat",
    "Pickup Long",
    "Destination Lat",
    "Destination Long",
    "Order No",
]

to_categorical = [
    "User Id",
    "Personal or Business",
    "Rider Id",
    "Placement - Day of Month",
    "Placement - Weekday (Mo = 1)",
    "Confirmation - Day of Month",
    "Confirmation - Weekday (Mo = 1)",
    "Arrival at Pickup - Day of Month",
    "Arrival at Pickup - Weekday (Mo = 1)",
    "Pickup - Day of Month",
    "Pickup - Weekday (Mo = 1)",
]

order_index = ["Order No"]

mean_encoded = ["hour_Pickup ", "hour_Placement "]
