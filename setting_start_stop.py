# import cv2
# import numpy as np

# class VideoControls:
#     def __init__(self):
#         self.is_running = False
#         self.button_height = 50
#         self.button_width = 100
#         self.margin = 10
        
#         # Create control panel window
#         self.panel = np.ones((70, 220, 3), dtype=np.uint8) * 240  # Light gray background
#         cv2.namedWindow('Controls')
#         cv2.setMouseCallback('Controls', self.handle_click)
        
#         self.update_panel()
        
#     def update_panel(self):
#         panel = self.panel.copy()
        
#         # Draw Start button
#         start_color = (200, 200, 200) if self.is_running else (144, 238, 144)  # Light green when not running
#         cv2.rectangle(panel, (self.margin, self.margin), 
#                      (self.margin + self.button_width, self.margin + self.button_height), 
#                      start_color, -1)
#         cv2.putText(panel, 'Start', (self.margin + 30, self.margin + 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
#         # Draw Stop button
#         stop_color = (144, 144, 238) if self.is_running else (200, 200, 200)  # Light red when running
#         cv2.rectangle(panel, (self.margin * 2 + self.button_width, self.margin),
#                      (self.margin * 2 + self.button_width * 2, self.margin + self.button_height),
#                      stop_color, -1)
#         cv2.putText(panel, 'Stop', (self.margin * 2 + self.button_width + 30, self.margin + 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
#         cv2.imshow('Controls', panel)
        
#     def handle_click(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if (self.margin <= x <= self.margin + self.button_width and 
#                 self.margin <= y <= self.margin + self.button_height):
#                 # Start button clicked
#                 self.is_running = True
#                 self.update_panel()
#             elif (self.margin * 2 + self.button_width <= x <= self.margin * 2 + self.button_width * 2 and 
#                   self.margin <= y <= self.margin + self.button_height):
#                 # Stop button clicked
#                 self.is_running = False
#                 self.update_panel()

#     def destroy(self):
#         cv2.destroyWindow('Controls')
import cv2
import numpy as np

class VideoControls:
    def __init__(self):
        self.is_running = False
        self.button_height = 50
        self.button_width = 100
        self.margin = 10
        
        # Create control panel window
        self.panel = np.ones((70, 220, 3), dtype=np.uint8) * 240
        cv2.namedWindow('Controls')
        cv2.setMouseCallback('Controls', self.handle_click)
        
        self.update_panel()
        
    def update_panel(self):
        panel = self.panel.copy()
        
        # Draw Start button
        start_color = (200, 200, 200) if self.is_running else (144, 238, 144)
        cv2.rectangle(panel, (self.margin, self.margin), 
                     (self.margin + self.button_width, self.margin + self.button_height), 
                     start_color, -1)
        cv2.putText(panel, 'Start', (self.margin + 30, self.margin + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw Stop button
        stop_color = (144, 144, 238) if self.is_running else (200, 200, 200)
        cv2.rectangle(panel, (self.margin * 2 + self.button_width, self.margin),
                     (self.margin * 2 + self.button_width * 2, self.margin + self.button_height),
                     stop_color, -1)
        cv2.putText(panel, 'Stop', (self.margin * 2 + self.button_width + 30, self.margin + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imshow('Controls', panel)
        
    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.margin <= x <= self.margin + self.button_width and 
                self.margin <= y <= self.margin + self.button_height):
                # Start button clicked
                self.is_running = True
                self.update_panel()
            elif (self.margin * 2 + self.button_width <= x <= self.margin * 2 + self.button_width * 2 and 
                  self.margin <= y <= self.margin + self.button_height):
                # Stop button clicked
                self.is_running = False
                self.update_panel()

    def destroy(self):
        cv2.destroyWindow('Controls')