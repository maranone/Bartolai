import pygame
import os
import numpy as np
import collections
from collections import deque, defaultdict

class PygameRenderer:
    def __init__(self, env):
        self.env = env
        # Constants and configuration
        self.width, self.height = 1280, 720
        self.genesis_width, self.genesis_height = (320 * 2), (224 * 2)
        self.game_x = self.width - self.genesis_width
        self.game_y = 0
        # Initialize additional attributes
        self.bar_spacing = 10
        self.max_bar_height = self.height - self.genesis_height - 40
        self.chart_height = (((self.height // 4) * 1) - 45) // 7
        self.chart_width = (self.width - self.genesis_width) // 2 - 10
        # Calculate the available width for bars
        self.available_width = self.genesis_width - 10 - 10  # 10 pixels margin from left edge
        # Starting x-position
        self.start_x = 10 + (self.width - self.genesis_width)  # 10 pixels margin from left edge
        self.end_x = self.width - 10
        # Pre-calculate rectangles and positions
        self.genesis_rect = pygame.Rect(self.width - self.genesis_width, 0, self.genesis_width, self.genesis_height)
        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        # Button sizes
        self.button_size = 30
        self.arrow_button_width = 60
        self.arrow_button_height = 60
        self.action_button_radius = 40
        # Color palette for charts
        self.color_palette = [
            (255, 87, 51), (255, 195, 0), (0, 230, 118),
            (41, 121, 255), (170, 0, 255), (255, 140, 0),
            (0, 230, 230)
        ]
        # Action mappings
        self.actions = {
            0: "B", 1: "A", 8: "C", 4: "up",
            5: "down", 6: "left", 7: "right"
        }
         # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        self.font = pygame.font.SysFont('Courier', 20)
        self.legend_font = pygame.font.SysFont('Courier', 18)
        # Data structures
        self.fill_percentages = {
            "B": 0, "A": 0, "C": 0, "up": 0,
            "down": 0, "left": 0, "right": 0
        }
        self.button_press_history = deque(maxlen=10)
        self.history_length = 2000
        self.histories = [collections.deque(maxlen=self.history_length) for _ in range(2)]


    def calculate_statistics(self, data):
        stats = {}
        for key, values in data.items():
            values = np.array(values)
            q1 = np.percentile(values, 25)
            median = np.median(values)
            q3 = np.percentile(values, 75)
            mean = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            current = values[-1] if len(values) > 0 else None
            stats[key] = {
                'q1': q1,
                'median': median,
                'q3': q3,
                'mean': mean,
                'min': min_val,
                'max': max_val,
                'current': current
            }
        return stats


    def drawhistories(self, current_data):
        # Update histories
        for i, value in enumerate(current_data):
            self.histories[i].append(value)

        #chart_height = (((self.height // 4) * 1) - 45) // 7  # 10 pixels margin top and bottom
        chart_height = (((self.height // 4) * 1) - 45)  # 10 pixels margin top and bottom
        #chart_width = (width - genesis_width) - 10  # 10 pixels margin left and right
        chart_width = (self.width - self.genesis_width) // 2 - 10  # 10 pixels margin left and right

        for i, history in enumerate(self.histories):
            # Calculate y position for this chart
            #chart_y = (self.height // 4 * 3) + 10 + i * (chart_height + 5)
            chart_y = (self.height // 4 * 3) + 10 + 0 * (chart_height + 5)

            # Normalize the data
            if history:
                max_value = max(history)
                min_value = min(history)
                normalized = [(v - min_value) / (max_value - min_value) if max_value != min_value else 0.5 for v in
                                history]
            else:
                normalized = []

            # Draw the chart
            for j in range(1, len(normalized)):
                x1 =  10 + (j - 1) * chart_width / self.history_length
                x2 =  10 + j * chart_width / self.history_length
                y1 = chart_y + chart_height - (normalized[j - 1] * chart_height)
                y2 = chart_y + chart_height - (normalized[j] * chart_height)
                pygame.draw.line(self.screen, self.color_palette[i], (x1, y1), (x2, y2), 2)

    def display_table(self, stats, width, height, previous_stats=None):
        font = pygame.font.SysFont('Arial', 20)
        small_font = pygame.font.SysFont('Arial', 16)
        bold_font = pygame.font.SysFont('Arial', 16, bold=True)

        background_color = (0, 0, 0)  # Black
        text_color = (255, 255, 255)  # White
        header_color = (100, 100, 100)  # Dark Gray
        row_alt_color = (20, 20, 20)  # Very Dark Gray

        # Define table structure
        col_widths = [200, 100, 100, 100, 100]
        row_height = 30
        table_width = sum(col_widths)
        table_height = (len(stats) + 1) * row_height

        # Calculate starting position to center the table
        start_x = (width - table_width) // 2
        start_y = (height - table_height) // 2

        # Draw table headers
        headers = ["Key", "Current", "Mean", "Min", "Max"]
        x = start_x
        for i, header in enumerate(headers):
            pygame.draw.rect(self.screen, header_color, (x, start_y, col_widths[i], row_height))
            text = font.render(header, True, text_color)
            self.screen.blit(text, (x + 5, start_y + 5))
            x += col_widths[i]

        # Draw table rows
        y = start_y + row_height
        for index, (key, stat) in enumerate(stats.items()):
            x = start_x
            row_color = row_alt_color if index % 2 else background_color

            # Determine color and arrow for current value
            current_color = text_color
            arrow = "→"
            if previous_stats and key in previous_stats:
                if stat['current'] > previous_stats[key]['current']:
                    current_color = (0, 255, 0)  # Green
                    arrow = "↑"
                elif stat['current'] < previous_stats[key]['current']:
                    current_color = (255, 0, 0)  # Red
                    arrow = "↓"

            row_data = [
                key,
                f"{self.smart_format(stat['current'])}",
                f"{self.smart_format(stat['mean'])}",
                f"{self.smart_format(stat['min'])}",
                f"{self.smart_format(stat['max'])}"
            ]

            for i, data in enumerate(row_data):
                pygame.draw.rect(self.screen, row_color, (x, y, col_widths[i], row_height))
                pygame.draw.rect(self.screen, text_color, (x, y, col_widths[i], row_height), 1)

                if i == 1:  # Current column
                    text = bold_font.render(data, True, current_color)
                else:
                    text = small_font.render(data, True, text_color)

                self.screen.blit(text, (x + 5, y + 5))
                x += col_widths[i]
            y += row_height


        return stats  # Return the current stats to be used as previous_stats in the next frame

    def smart_format(self, value):
        # If the value is an integer, return it as an integer string
        if value == int(value):
            return f"{int(value)}"

        # Convert the value to a string to find significant figures
        str_value = f"{value:.10f}".rstrip('0').rstrip('.')

        # Find the position of the first significant figure after the decimal point
        decimal_pos = str_value.find('.')
        first_significant_pos = decimal_pos + 1
        while first_significant_pos < len(str_value) and str_value[first_significant_pos] == '0':
            first_significant_pos += 1

        # Calculate how many decimal places to show, ensuring at least 2 significant figures after the first non-zero digit
        num_decimals = first_significant_pos - decimal_pos + 1
        decimals_to_show = max(3, num_decimals)

        # Format the number with the determined number of decimal places
        formatted_value = f"{value:.{decimals_to_show}f}".rstrip('0').rstrip('.')

        return formatted_value

    def read_log_file(self, file_path):
        log_data = {}
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    pairs = line.split(',')
                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip()
                            value = value.strip().strip('"')  # Strip trailing quotes
                            try:
                                numeric_value = float(value)
                                if key in log_data:
                                    log_data[key].append(numeric_value)
                                else:
                                    log_data[key] = [numeric_value]
                            except ValueError:
                                print(f"Skipping non-numeric value: {value} for key: {key}")
        except FileNotFoundError:
            #print(f"Log file not found: {file_path}")
            return log_data
        except Exception as e:
            #print(f"Error reading log file: {e}")
            return log_data
        return log_data


    def drawloggertable(self):
        file_path = os.path.join('./logs/training_log.csv')
        try:
            log_data = self.read_log_file(file_path)
            stats = self.calculate_statistics(log_data)
            # Use log_data as before
        except Exception as e:
            print(f"Error processing log file: {e}")
            log_data = {}  # Use an empty dict if there's an error

        if log_data != {}:
            window_width = (self.width-self.genesis_width)
            window_height = (self.height // 4) * 3
            #display_plots(stats, window_width, window_height)
            stats = self.calculate_statistics(log_data)
            if self.previous_stats == None:
                self.previous_stats = stats

            if self.previous_stats_temp != stats:
                self.previous_stats = self.previous_stats_temp
                self.previous_stats_temp = stats

            temp_stats_temp = self.display_table(stats, window_width, window_height, self.previous_stats)


    def drawcontroller(self):
        # Define your new width and calculate the scaling factor
        new_width = (self.width - self.genesis_width) // 2
        scaling_factor = new_width / (self.width - self.genesis_width)

        x = (self.width - self.genesis_width) // 2
        y = (self.height // 4) * 3
        box_width = (self.width - self.genesis_width) * scaling_factor
        box_height = (self.height - self.genesis_height) * scaling_factor
        button_size = int(30 * scaling_factor)
        arrow_button_width = int(60 * scaling_factor)
        arrow_button_height = int(60 * scaling_factor)
        action_button_radius = int(40 * scaling_factor)

        left_arrow_rect = pygame.Rect(
            x + int(50 * scaling_factor),
            y + box_height // 2 - arrow_button_height // 2,
            arrow_button_width,
            arrow_button_height
        )
        up_arrow_rect = pygame.Rect(
            x + int(120 * scaling_factor),
            y + box_height // 2 - arrow_button_height - int(10 * scaling_factor),
            arrow_button_height,
            arrow_button_width
        )
        right_arrow_rect = pygame.Rect(
            x + int(50 * scaling_factor) + arrow_button_width + int(80 * scaling_factor),
            y + box_height // 2 - arrow_button_height // 2,
            arrow_button_width,
            arrow_button_height
        )
        down_arrow_rect = pygame.Rect(
            x + int(120 * scaling_factor),
            y + box_height // 2 + int(10 * scaling_factor),
            arrow_button_height,
            arrow_button_width
        )

        A_button_center = (x + box_width // 2 + int(50 * scaling_factor), y + box_height // 2)
        B_button_center = (x + box_width // 2 + int(150 * scaling_factor), y + box_height // 2)
        C_button_center = (x + box_width // 2 + int(250 * scaling_factor), y + box_height // 2)

        # Iterate through actions and append one action at a time based on the condition
        for i in range(len(self.curac)):
            if i in self.actions and self.curac[i] == 1:
                self.button_press_history.append([self.actions[i]])  # Wrap in a list

        # print(button_press_history)

        self.update_fill_percentages(self.button_press_history)

        pygame.draw.rect(self.screen, self.white, (x, y, box_width, box_height), 2)


        # Then use it like this:
        self.draw_filled_rect(left_arrow_rect, self.fill_percentages["left"],
                            PygameRenderer.get_gradient_color(self.red, self.fill_percentages["left"]))
        self.draw_filled_rect(up_arrow_rect, self.fill_percentages["up"], PygameRenderer.get_gradient_color(self.green, self.fill_percentages["up"]))
        self.draw_filled_rect(right_arrow_rect, self.fill_percentages["right"],
                            PygameRenderer.get_gradient_color(self.blue, self.fill_percentages["right"]))
        self.draw_filled_rect(down_arrow_rect, self.fill_percentages["down"],
                            PygameRenderer.get_gradient_color(self.red, self.fill_percentages["down"]))

        self.draw_filled_circle(A_button_center, action_button_radius, self.fill_percentages["A"],
                            PygameRenderer.get_gradient_color(self.blue, self.fill_percentages["A"]))
        self.draw_filled_circle(B_button_center, action_button_radius, self.fill_percentages["B"],
                            PygameRenderer.get_gradient_color(self.green, self.fill_percentages["B"]))
        self.draw_filled_circle(C_button_center, action_button_radius, self.fill_percentages["C"],
                            PygameRenderer.get_gradient_color(self.red, self.fill_percentages["C"]))

        text = self.font.render(f"U: {self.steps_for_log}", True, (255, 255, 255))
        text_rect = text.get_rect(left=100 + (self.width - self.genesis_width) // 2, bottom=self.height - 10)
        self.screen.blit(text, text_rect)


    @staticmethod
    def get_gradient_color(base_color, percentage):
            return tuple(int(255 - (255 - c) * (percentage / 100)) for c in base_color)

    def drawresetcount(self):
        text = self.font.render(f"R: {self.reset_count}", True, (255, 255, 255))
        text_rect = text.get_rect(left=10 + (self.width - self.genesis_width) // 2, bottom=self.height - 10)
        self.screen.blit(text, text_rect)


    def drawlegends(self, screen, legend_texts):
        # Setup for the legend
        legend_font = pygame.font.SysFont('Courier', 18)  # Monospaced font for even spacing
        legend_box_size = 25
        legend_y = self.height - 60  # 30 pixels from the bottom of the screen
        # Calculate available width for the legend
        legend_width = self.width - (self.width - self.genesis_width)
        num_items = len(legend_texts)
        # Calculate spacing between items
        item_width = legend_width / num_items
        for i, text in enumerate(legend_texts):
            # Calculate position for each item
            box_x = (self.width - self.genesis_width) + (i * item_width)
            text_x = box_x + legend_box_size + 5
            # Draw the text
            lines = text.splitlines()  # Split text into lines based on '\n'
            for j, line in enumerate(lines):
                text_surface = legend_font.render(line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(left=text_x,
                                                    top=legend_y + j * legend_box_size)  # Adjust vertical position for each line
                screen.blit(text_surface, text_rect)

    def drawbars(self, screen, bar_data, start_x, color_palette, font):
        # Calculate bar width
        num_bars = len(bar_data)
        total_spacing = self.bar_spacing * (num_bars - 1)
        bar_width = (self.available_width - total_spacing) // num_bars

        for i, (current_value, max_value) in enumerate(bar_data):
            # Calculate the bar height as a percentage of the maximum value
            if max_value != 0:
                bar_percentage = current_value / max_value
            else:
                bar_percentage = 0

            bar_height = int(bar_percentage * self.max_bar_height)

            # Calculate bar position
            bar_x = start_x + i * (bar_width + self.bar_spacing)
            bar_y = (self.height - self.genesis_height) + (self.genesis_height - bar_height)   # Position from bottom of screen

            # Draw the bar with color from the palette
            bar_color = color_palette[i % len(color_palette)]
            pygame.draw.rect(screen, bar_color, (bar_x, bar_y, bar_width, bar_height))

            # Draw the outline of the bar
            pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)

            # Draw the value on top of each bar
            percentage = round((current_value * 100) / (max_value + 0.0001))
            text = font.render(f"{percentage}%", True, (255, 255, 255))
            text_rect = text.get_rect(centerx=bar_x + bar_width // 2, bottom=bar_y - 5)
            screen.blit(text, text_rect)
        
    def update_fill_percentages(self, button_press_history):
        total_presses = len(button_press_history)
        if total_presses == 0:
            # If no presses recorded, set all percentages to zero
            for button in self.fill_percentages:
                self.fill_percentages[button] = 0
        else:
            # Count occurrences of each button press
            button_counts = defaultdict(int)
            for step in button_press_history:
                for button in step:
                    button_counts[button] += 1

            # Calculate percentages based on counts
            for button in self.fill_percentages:
                self.fill_percentages[button] = (button_counts[button] / total_presses) * 100
        
    def draw_filled_rect(self, rect, percentage, color):
        pygame.draw.rect(self.screen, self.white, rect, 2)
        filled_rect_height = int(rect.height * (percentage / 100))
        top_position = rect.bottom - filled_rect_height  # Start filling from the bottom upwards
        filled_rect = pygame.Rect(rect.x, top_position, rect.width, filled_rect_height)
        pygame.draw.rect(self.screen, color, filled_rect)

    def draw_filled_circle(self, center, radius, percentage, color):
        pygame.draw.circle(self.screen, self.white, center, radius, 2)
        filled_circle_radius = int(radius * (percentage / 100))
        pygame.draw.circle(self.screen, color, center, filled_circle_radius)  

    def display_policy(self, info):
        enemy_count = sum(1 for i in range(1, 8) if info[f'enemy{i}'] > 0)
        rect_x = self.width - 267
        rect_width = 260
        rect_height = 52  # Define the height of the rectangle
        top_position = 5  # Define the top position of the rectangle
        text_color = (255, 255, 255)  # White text color
        background_color = (0, 0, 0)  # Black background
        enemy_color = (0, 255, 0)  # Green color for enemies
        
        # Background rectangle
        pygame.draw.rect(self.screen, background_color, pygame.Rect(rect_x, top_position, rect_width, rect_height))  # Black background

        if enemy_count > 0:
            # Draw green rectangle
            pygame.draw.rect(self.screen, enemy_color, pygame.Rect(rect_x, top_position, rect_width, rect_height))  
            text = "Enemy"
        else:
            text = "Exploration"

        # Draw border rectangle
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(rect_x, top_position, rect_width, rect_height), 2)  # White border
        
        # Font settings
        font_size = int(rect_height / 1.5)  # Adjust font size relative to rectangle height
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=(rect_x + rect_width / 2, top_position + rect_height / 2))
        
        # Draw text
        self.screen.blit(text_surface, text_rect)


    def render(self, render_data):

        info = render_data['info']
        self.steps_without_reward = render_data['steps_without_reward']
        self.max_steps_without_reward = render_data['max_steps_without_reward']
        self.cumulative_health = render_data['cumulative_health']
        self.max_hea = render_data['max_hea']
        self.cumulative_reward = render_data['cumulative_reward']
        self.max_reward = render_data['max_reward']
        self.cumulative_score = render_data['cumulative_score']
        self.max_score = render_data['max_score']
        self.cumulative_map = render_data['cumulative_map']
        self.max_map = render_data['max_map']
        self.cumulative_damage = render_data['cumulative_damage']
        self.max_dam = render_data['max_dam']
        self.reset_count = render_data['reset_count']
        self.curac = render_data['curac']
        self.steps_for_log = render_data['steps_for_log']
        self.previous_stats = render_data['previous_stats']
        self.previous_stats_temp = render_data['previous_stats_temp']

        self.screen.fill(self.black)
        pygame.draw.rect(self.screen, self.white, self.genesis_rect, 1)
        raw_frame = self.env.render()
        game_surface = pygame.surfarray.make_surface(raw_frame.swapaxes(0, 1))
        game_surface = pygame.transform.scale(game_surface, (self.genesis_width, self.genesis_height))
        self.screen.blit(game_surface, (self.game_x, self.game_y))


        # Your data
        bar_data = [
            [self.max_steps_without_reward - self.steps_without_reward, self.max_steps_without_reward],
            [(info['time']),40],
            [abs(self.max_hea) - abs(self.cumulative_health), abs(self.max_hea)],
            [self.cumulative_reward, self.max_reward],
            [self.cumulative_score, self.max_score],
            [self.cumulative_map, self.max_map],
            [self.cumulative_damage, self.max_dam]
        ]

        # Draw the bars
        self.drawbars(self.screen, bar_data, self.start_x, self.color_palette, self.font)


        # Define your legend texts
        legend_texts = ['SWR\n' + str(round(self.steps_without_reward)), 'TIM\n' + str(round(info['time'])), 'HEA\n' + str(round(self.cumulative_health,2)),
                        'REW\n' + str(round(self.cumulative_reward, 3)), 'SCO\n' + str(round(info['score'],2)), 'MAP\n' + str(round(self.cumulative_map,2)),
                        'DAM\n' + str(round(self.cumulative_damage,2))]

        self.drawlegends(self.screen, legend_texts)

        self.drawresetcount()

        current_data = [
            #self.max_steps_without_reward - self.steps_without_reward,
            #info['time'],
            #abs(self.max_hea) - abs(self.cumulative_health),
            self.cumulative_reward,
            self.cumulative_score,
            #self.cumulative_map,
            #self.cumulative_damage
        ]

        self.drawhistories(current_data)
        self.drawcontroller()
        self.drawloggertable()
        self.display_policy(info)


        pygame.display.flip()
        

        