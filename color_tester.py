import cv2 as cv
import numpy as np
import pygame
import sys

# --- Pygame and window initialization ---
pygame.init()
WIDTH, HEIGHT = 1100, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("HSV Color Detector (Light to Dark) - Pygame")

FONT = pygame.font.SysFont("Arial", 20)
SMALL_FONT = pygame.font.SysFont("Arial", 16)

# Initial HSV ranges for each color (from light to dark)
# H: 0-179, S: 0-255, V: 0-255
rangos = {
    "Negro": [[0, 0, 0], [179, 255, 60]],           # Low V for black
    "Azul": [[90, 80, 40], [130, 255, 255]],        # Blue
    "Verde": [[35, 80, 40], [85, 255, 255]],        # Green
    "Rojo": [[0, 100, 40], [10, 255, 255]],         # Red (low H range)
    "Morado": [[130, 80, 40], [160, 255, 255]],     # Purple
}
colores = list(rangos.keys())
color_seleccionado = 1  # Selected color index (Blue by default)

# Sliders: [H low, H high, S low, S high, V low, V high]
slider_labels = ["H low", "H high", "S low", "S high", "V low", "V high"]
slider_ranges = [(0, 179), (0, 179), (0, 255), (0, 255), (0, 255), (0, 255)]
# Current slider values (updated when color changes)
slider_values = rangos[colores[color_seleccionado]][0] + rangos[colores[color_seleccionado]][1]
slider_rects = []

def draw_slider(x, y, w, h, min_val, max_val, value, label):
    """
    Draws a horizontal slider on the pygame screen.
    Allows clicking anywhere on the bar to set the value.

    Args:
        x, y: slider position
        w, h: slider width and height
        min_val, max_val: slider min and max values
        value: current value
        label: slider label

    Returns:
        pygame.Rect of the slider bar area
    """
    # Draw bar
    pygame.draw.rect(screen, (200, 200, 200), (x, y, w, h), border_radius=6)
    # Draw handle
    pos = int((value - min_val) / (max_val - min_val) * w)
    pygame.draw.rect(screen, (100, 100, 255), (x + pos - 7, y - 5, 14, h + 10), border_radius=7)
    # Draw label and value
    txt = SMALL_FONT.render(f"{label}: {value}", True, (0, 0, 0))
    screen.blit(txt, (x + w + 15, y))
    return pygame.Rect(x, y, w, h)  # The rect covers the whole bar

def update_slider_from_mouse(mx, my):
    """
    Updates the slider value if the user clicks anywhere on the bar.

    Args:
        mx, my: mouse position
    """
    for i, rect in enumerate(slider_rects):
        if rect.collidepoint(mx, my):
            x, y, w, h = rect
            min_val, max_val = slider_ranges[i]
            rel_x = mx - x
            rel_x = max(0, min(w, rel_x))
            value = int(min_val + (rel_x / w) * (max_val - min_val))
            slider_values[i] = value
            # Update the selected color's range
            rangos[colores[color_seleccionado]] = [
                slider_values[:3], slider_values[3:]
            ]
            break

def draw_color_selector():
    """
    Draws the color selection buttons.

    Returns:
        List of pygame.Rect for each color button.
    """
    rects = []
    for idx, color in enumerate(colores):
        rect = pygame.Rect(60, 400 + idx * 48, 140, 40)
        pygame.draw.rect(screen, (180, 180, 255) if idx == color_seleccionado else (220, 220, 220), rect, border_radius=8)
        txt = FONT.render(color, True, (0, 0, 0))
        screen.blit(txt, (rect.x + 15, rect.y + 8))
        rects.append(rect)
    return rects

def opencv_mask_frame(frame):
    """
    Applies the selected HSV range and returns the binary mask.

    Args:
        frame: OpenCV BGR image

    Returns:
        Binary mask (np.ndarray)
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    bajo, alto = rangos[colores[color_seleccionado]]
    bajo_np = np.array(bajo, dtype=np.uint8)
    alto_np = np.array(alto, dtype=np.uint8)
    mask = cv.inRange(hsv, bajo_np, alto_np)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)
    return mask

def cvimg_to_pygame(img):
    """
    Converts an OpenCV BGR image to a Pygame Surface.

    Args:
        img: OpenCV BGR image

    Returns:
        pygame.Surface
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.rot90(img)
    surf = pygame.surfarray.make_surface(img)
    return surf

def mask_to_pygame(mask):
    """
    Converts a binary mask to a Pygame Surface.

    Args:
        mask: binary mask (np.ndarray)

    Returns:
        pygame.Surface
    """
    mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    mask_rgb = np.rot90(mask_rgb)
    surf = pygame.surfarray.make_surface(mask_rgb)
    return surf

def update_sliders_from_color():
    """
    Updates the slider values when a new color is selected.
    """
    global slider_values
    slider_values = rangos[colores[color_seleccionado]][0] + rangos[colores[color_seleccionado]][1]

# --- Main loop ---
cap = cv.VideoCapture(0)
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            # Sliders
            update_slider_from_mouse(mx, my)
            # Color selector
            color_rects = draw_color_selector()
            for idx, rect in enumerate(color_rects):
                if rect.collidepoint(mx, my):
                    color_seleccionado = idx
                    update_sliders_from_color()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                cap.release()
                pygame.quit()
                sys.exit()

    # Capture frame from camera and process
    ret, frame = cap.read()
    if not ret:
        continue

    frame_small = cv.resize(frame, (340, 260))
    mask = opencv_mask_frame(frame_small)

    screen.fill((240, 240, 240))

    # Sliders (aligned to the left)
    slider_rects = []
    for i, (label, (min_val, max_val), value) in enumerate(zip(slider_labels, slider_ranges, slider_values)):
        rect = draw_slider(60, 60 + i * 48, 380, 22, min_val, max_val, value, label)
        slider_rects.append(rect)

    # Color selector (bottom left)
    draw_color_selector()

    # Instructions
    txt = SMALL_FONT.render("Click on the sliders to adjust the HSV range. Click on the color to select it.", True, (0, 0, 0))
    screen.blit(txt, (60, 20))
    txt2 = SMALL_FONT.render("Press ESC or close the window to exit.", True, (0, 0, 0))
    screen.blit(txt2, (60, 40))

    # Show original frame and mask (centered)
    surf_frame = cvimg_to_pygame(frame_small)
    surf_mask = mask_to_pygame(mask)
    screen.blit(surf_frame, (500, 60))
    screen.blit(surf_mask, (850, 60))
    pygame.draw.rect(screen, (0, 0, 0), (500, 60, 340, 260), 2)
    pygame.draw.rect(screen, (0, 0, 0), (850, 60, 340, 260), 2)
    screen.blit(SMALL_FONT.render("Original", True, (0,0,0)), (500, 40))
    screen.blit(SMALL_FONT.render("Mask", True, (850, 40)))

    pygame.display.flip()
    clock.tick(30)
