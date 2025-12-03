from . import BaseController
import numpy as np
import math

class Controller(BaseController):
    
    def __init__(self):

        self.p = 0.285
        self.i = 0.0873
        self.d = -0.1
        
        # 🔧 State Variables
        self.error_integral = 0
        self.prev_error = 0
        
 
        self.prev_steer_command = 0.0 # Stores command from the previous step
        self.max_steer_rate = 0.25 

        # 📉 D-Term Filtering (New Filter Variables)
        self.D_filter_gain = 0.5 
        self.prev_filtered_error = 0.0
        
        # ⚙️ Feedforward Parameters
        self.steer_factor = 13.55
        self.steer_sat_v = 20
        self.steer_command_sat = 2
        self.counter = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.counter += 1
        if self.counter == 81:
            self.error_integral = 0
            self.prev_error = 0
            self.prev_steer_command = 0.0 # Reset command history
            self.prev_filtered_error = 0.0 # Reset filter history


        # Optimized future lataccel average calculation
        if len(future_plan.lataccel) >= 3:
            lataccel_combined = np.array([target_lataccel] + future_plan.lataccel[:3])
            # Original: weights = np.array([5, 5, 5, 5])
            weights = np.array([5, 5, 5, 4]) 
            target_lataccel = np.average(lataccel_combined, weights=weights)

        # --- PID Error Calculation with D-Term Filtering ---
        error = target_lataccel - current_lataccel
        self.error_integral += error

        # Smooths the error signal used for the derivative calculation (error_diff)
        filtered_error = self.D_filter_gain * error + (1.0 - self.D_filter_gain) * self.prev_filtered_error
        
        # Calculate error difference using the filtered error
        error_diff = filtered_error - self.prev_filtered_error
        self.prev_filtered_error = filtered_error # Store the filtered value
        
        # Store raw error for proportional term in the next step
        self.prev_error = error 

        # Dynamic scaling for high magnitude lataccel
        pid_factor = max(0.5, 1 - 0.23 * abs(target_lataccel))

        # Proportional gain dynamically adjusted for acceleration
        p_dynamic = max(0.1, self.p - 0.1 * abs(state.a_ego))

        # PID Control signal
        # Note: P and I terms use the raw 'error', D term uses the filtered error difference
        u_pid = (p_dynamic * error + self.i * self.error_integral + self.d * error_diff) * pid_factor

        # Feedforward control: Adjusted with a sigmoid function for smoothness
        steer_accel_target = target_lataccel - state.roll_lataccel
        steer_command = steer_accel_target * self.steer_factor / max(self.steer_sat_v, state.v_ego)
        steer_command = 2 * self.steer_command_sat / (1 + math.exp(-steer_command)) - self.steer_command_sat

        # Combined control signal with feedforward gain
        u_ff = 0.8 * steer_command
        
        # Calculate the base new command
        new_command = u_pid + u_ff
        
        # Limits the change in command between steps to reduce jerk
        rate_limited_command = np.clip(
            new_command,
            self.prev_steer_command - self.max_steer_rate,
            self.prev_steer_command + self.max_steer_rate
        )

        # Apply final saturation clip (-2 to 2)
        final_command = np.clip(rate_limited_command, -2, 2)
        
        # Store the final command for the next iteration's rate limit check
        self.prev_steer_command = final_command

        return final_command