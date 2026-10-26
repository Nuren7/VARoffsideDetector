import cv2
import numpy as np
from typing import List, Dict, Optional, Any


class OffsideAnalyzer:
  """for calculating the geometry of offside line"""
  
  def __init__(self, output_video_path: str = "output_offside.mp4", fps: int = 30.0):
    self.output_video_path = output_video_path
    self.fps = fps
    self.video_writer: Optional[cv2.VideoWriter] = None
    
  def process_and_draw(self, input_video_path: str, video_data: List[Dict[str, Any]], player_teams: List[Dict[int, int]]) -> None:
    """Process the video and draw the offside line based on player positions and teams."""
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
      raise ValueError(f"Could not open video file: {input_video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
      fps = self.fps  # fallback to default fps if unable to get from video
      
    #avc1 for broader compatibility, mp4v for mp4
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # or "mp4v" for .mp4
    out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # fallback to mp4v
      out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
    
    frame_index = 0

    try:
      while cap.isOpened():
        success, frame = cap.read()

        if not success:
          break
        
        #get the current frame data
        if frame_index < len(video_data):
          current_frame_data = video_data[frame_index]
          players = current_frame_data.get("players", [])
          ball = current_frame_data.get("ball")
          
          #Get the team for this frame 
          teams_map = player_teams[frame_index] if frame_index < len(player_teams) else {}
          
          #example of how to draw the offside line based on player positions and teams
          
          #find defending player to draw the offside line
          defending_players = []
          attacking_players = []
          
          for i, player in enumerate(players):
            bbox = player["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            team_id = teams_map.get(i, -1)
            
            #chose colo based on team
            color = (0,0,255) if team_id == 0 else (255,0,0) if team_id == 1 else (0,255,255)
          
            #draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if team_id == 0:  # Assuming team_id 0 is the defending team
              #save to the right
              defending_players.append(x2)
            elif team_id == 1:  # Assuming team_id 1 is the attacking team
              #save to the left
              attacking_players.append(x1)

          #find the second last defender if there is enough players
          if len(defending_players) >= 2:
            #sort x-coords
            defending_players.sort(reverse=True)
            #second last defender
            second_last_defender_x = defending_players[1]
            
            #draw the offside line
            cv2.line(frame, (second_last_defender_x, 0), (second_last_defender_x, height), (0, 255, 255), 2)
            cv2.putText(frame, "Offside Line", (second_last_defender_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            
            
          #draw the ball if present
          if ball:
            bx1, by1, bx2, by2 = map(int, ball["bbox"])
            cv2.circle(frame, (int((bx1 + bx2) /2 ), int((by1 + by2) / 2)), 8, (0,255,0), -1)

        out.write(frame)
        frame_index += 1

    finally:
      cap.release()
      out.release()
      print(f"Output video saved to {self.output_video_path}")
        