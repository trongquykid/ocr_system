import json

def open_json(input_path):
    with open(input_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_json(output_path, data):
    with open(output_path, 'w+', encoding='utf-8') as file:
        json.dump(data, file)

def get_title(title_name, text_lines):
    content_polygon = None
    for line in text_lines:
        if title_name in line["text"]:
            content_polygon = line["polygon"]
            return content_polygon

def get_content(datas, name, title, result):
    no_number = None
    text_lines = datas[name][0]["text_lines"]

    content_no_polygon = get_title(title, text_lines)

    if content_no_polygon:
        content_no_bottom_y = max(point[1] for point in content_no_polygon)
        content_no_left_x = min(point[0] for point in content_no_polygon)
        # content_no_right_x = max(point[0] for point in content_no_polygon)
        possible_numbers = []
        
        for line in text_lines:
            line_top_y = min(point[1] for point in line["polygon"])
            line_left_x = min(point[0] for point in line["polygon"])
            line_right_x = max(point[0] for point in line["polygon"])
            
            # if line_top_y > content_no_bottom_y and line_left_x >= content_no_left_x and line_right_x <= content_no_right_x:
            if line_top_y > content_no_bottom_y and line_left_x >= content_no_left_x :
                possible_numbers.append((line_top_y, line["text"]))
        
        if possible_numbers:
            
            possible_numbers.sort()  # Sort by top Y coordinate
            no_number = possible_numbers[0][1]
    
    result[title] = no_number
    return result