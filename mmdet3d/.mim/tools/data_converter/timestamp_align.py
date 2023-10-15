from datetime import datetime, timedelta

def timestamp_to_beijing_time(timestamp):
    dt_object = datetime(1970, 1, 1) + timedelta(microseconds=int(timestamp) // 1000)
    beijing_time = dt_object + timedelta(hours=8)
    beijing_time_str = beijing_time.strftime('%Y%m%d%H%M%S%f')
    beijing_time_str = beijing_time_str[0:15]
    # beijing_time_str = beijing_time.strftime('%Y%m%d%H%M%S')
    return beijing_time_str

def convert_first_column(input_file):
    converted_lines = []
    with open(input_file, 'r') as infile:
        for line in infile:
            data = line.strip().split(',')
            if len(data) > 0:
                timestamp = data[0]
                try:
                    timestamp_int = int(timestamp)
                    beijing_time = timestamp_to_beijing_time(timestamp_int)
                    data[0] = beijing_time  # Update the first element with the converted timestamp
                except ValueError:
                    # Handle the case where the timestamp cannot be converted to an integer
                    pass
                converted_line = ','.join(data) + '\n'
                converted_lines.append(converted_line)

    # Write the converted lines back to the original file
    with open(input_file, 'w') as outfile:
        outfile.writelines(converted_lines)

# Example usage
input_file = "/home/slj/Documents/workspace/ThomasVision/data/smart_lane/Odometry.txt"
convert_first_column(input_file)