import csv
import os
from stable_baselines3.common.callbacks import BaseCallback

class StatsLogger(BaseCallback):
    def __init__(self, verbose=0, csv_path='training_log.csv'):
        super(StatsLogger, self).__init__(verbose)
        self.csv_path = csv_path
        self.csv_initialized = False

        # Check if the file exists and if so, delete it to start fresh
        if os.path.isfile(self.csv_path):
            os.remove(self.csv_path)

    def _on_step(self):
        if self.n_calls % self.model.n_steps == 0:  # Log every n_steps
            # Collect data manually
            data = {
                'time/iterations': self.num_timesteps // self.model.n_steps,
                'time/total_timesteps': self.num_timesteps,
                'total_timesteps': self.num_timesteps,
                'iteration': self.n_calls // self.model.n_steps,
                'n_steps': self.model.n_steps,
                'idx': self.n_calls,
            }

            # Add training stats if available
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                for key, value in self.model.logger.name_to_value.items():
                    data[key] = value

            # Write the data to CSV
            self.write_to_csv(data)

        return True

    def write_to_csv(self, data):
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = ', '.join([f"{label}:{value}" for label, value in data.items()])
            writer.writerow([row])