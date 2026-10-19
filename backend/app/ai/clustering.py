import cv2
import numpy as np
from sklearn.cluster import KMeans

class Teamclassifier:
  """Team classifier with the help of k-means"""
  def __init__(self, n_clusters: int=3):
    self.n_clusters = n_clusters
    self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
    
  def extract_jersey_color(self, frame : np.ndarray, bbox: list) -> np.ndarray:
    """Cutting the upper half of the players"""
    
    #Count out the height and crop the upper half
    x1,y1,x2,y2 = map(int, bbox)
    height = y2-y1
    jersey_y2 = y1 + int(height * 0.5)
    
    #Ensure the cords is inside the dimensions
    h_frame, w_frame, _ = frame.shape
    x1 = max(0,x1)
    y1 = max(0,y1)
    x2 = min(w_frame,x2)
    jersey_y2 = min(h_frame, jersey_y2)
    
    #if the box is not good
    if x1 >= x2 or y1 >= jersey_y2:
      return np.array([])
    
    #crop the pic
    cropped_jersey = frame[y1:jersey_y2, x1:x2]
    return cropped_jersey
  
  def classify_player(self, frame: np.ndarray, bbox: list) -> int:
    """cropping the jersey, appy K-means on the colors and returning a team"""
    jersey = self.extract_jersey_color(frame, bbox)
    
    if jersey.size == 0:
      return -1
    
    #Convert BGR to RGB
    jersey_rgb = cv2.cvtColor(jersey, cv2.COLOR_BGR2RGB)
    
    #Format pictures t a list of pixles
    pixels = jersey_rgb.reshape(-1, 3)
    
    #if its not enough pixels
    if len(pixels) < self.n_clusters:
      return -1
    
    #Fit k-means to color of the jersey
    self.kmeans.fit(pixels)
    
    #find the the dominant color
    labels = self.kmeans.labels_
    counts = np.bincount(labels)
    dominant_cluster = np.argmax(counts)
    
    return int(dominant_cluster)
  