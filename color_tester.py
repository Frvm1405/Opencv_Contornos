import cv2 as cv
import numpy as np
import pygame
import sys

# --- Configuración inicial de Pygame y ventana ---
pygame.init()
WIDTH, HEIGHT = 1100, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Detector de Colores HSV (Claro a Oscuro) - Pygame")

FONT = pygame.font.SysFont("Arial", 20)
SMALL_FONT = pygame.font.SysFont("Arial", 16)

# Rangos HSV iniciales para cada color (de claro a oscuro)
# H: 0-179, S: 0-255, V: 0-255
rangos = {
    "Negro": [[0, 0, 0], [179, 255, 60]],           # V bajo para negros
    "Azul": [[90, 80, 40], [130, 255, 255]],        # Azul
    "Verde": [[35, 80, 40], [85, 255, 255]],        # Verde
    "Rojo": [[0, 100, 40], [10, 255, 255]],         # Rojo bajo
    "Morado": [[130, 80, 40], [160, 255, 255]],     # Morado
}
colores = list(rangos.keys())
color_seleccionado = 1  # Índice del color seleccionado (Azul por defecto)

# Sliders: [H bajo, H alto, S bajo, S alto, V bajo, V alto]
slider_labels = ["H bajo", "H alto", "S bajo", "S alto", "V bajo", "V alto"]
slider_ranges = [(0, 179), (0, 179), (0, 255), (0, 255), (0, 255), (0, 255)]
# Valores actuales de los sliders (se actualizan al cambiar de color)
slider_values = rangos[colores[color_seleccionado]][0] + rangos[colores[color_seleccionado]][1]
slider_rects = []

def draw_slider(x, y, w, h, min_val, max_val, value, label):
    """
    Dibuja un slider horizontal en la pantalla de pygame.
    Permite hacer click en cualquier parte de la barra para mover el valor.

    Args:
        x, y: posición del slider
        w, h: ancho y alto del slider
        min_val, max_val: valores mínimo y máximo del slider
        value: valor actual
        label: texto del slider

    Returns:
        pygame.Rect del área de la barra del slider
    """
    # Barra
    pygame.draw.rect(screen, (200, 200, 200), (x, y, w, h), border_radius=6)
    # Handle
    pos = int((value - min_val) / (max_val - min_val) * w)
    pygame.draw.rect(screen, (100, 100, 255), (x + pos - 7, y - 5, 14, h + 10), border_radius=7)
    # Etiqueta y valor
    txt = SMALL_FONT.render(f"{label}: {value}", True, (0, 0, 0))
    screen.blit(txt, (x + w + 15, y))
    return pygame.Rect(x, y, w, h)  # El rect es toda la barra

def update_slider_from_mouse(mx, my):
    """
    Permite mover el slider haciendo click en cualquier parte de la barra.

    Args:
        mx, my: posición del mouse
    """
    for i, rect in enumerate(slider_rects):
        if rect.collidepoint(mx, my):
            x, y, w, h = rect
            min_val, max_val = slider_ranges[i]
            rel_x = mx - x
            rel_x = max(0, min(w, rel_x))
            value = int(min_val + (rel_x / w) * (max_val - min_val))
            slider_values[i] = value
            # Actualiza el rango del color seleccionado
            rangos[colores[color_seleccionado]] = [
                slider_values[:3], slider_values[3:]
            ]
            break

def draw_color_selector():
    """
    Dibuja los botones de selección de color.

    Returns:
        Lista de pygame.Rect de cada botón de color.
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
    Aplica el rango HSV seleccionado y devuelve la máscara binaria.

    Args:
        frame: imagen BGR de OpenCV

    Returns:
        Máscara binaria (np.ndarray)
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
    Convierte una imagen BGR de OpenCV a una Surface de Pygame.

    Args:
        img: imagen BGR de OpenCV

    Returns:
        pygame.Surface
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.rot90(img)
    surf = pygame.surfarray.make_surface(img)
    return surf

def mask_to_pygame(mask):
    """
    Convierte una máscara binaria a una Surface de Pygame.

    Args:
        mask: máscara binaria (np.ndarray)

    Returns:
        pygame.Surface
    """
    mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    mask_rgb = np.rot90(mask_rgb)
    surf = pygame.surfarray.make_surface(mask_rgb)
    return surf

def update_sliders_from_color():
    """
    Actualiza los valores de los sliders cuando se selecciona un nuevo color.
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
            # Selector de color
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

    # Captura frame de la cámara y procesa
    ret, frame = cap.read()
    if not ret:
        continue

    frame_small = cv.resize(frame, (340, 260))
    mask = opencv_mask_frame(frame_small)

    screen.fill((240, 240, 240))

    # Sliders (más a la izquierda)
    slider_rects = []
    for i, (label, (min_val, max_val), value) in enumerate(zip(slider_labels, slider_ranges, slider_values)):
        rect = draw_slider(60, 60 + i * 48, 380, 22, min_val, max_val, value, label)
        slider_rects.append(rect)

    # Selector de color (abajo a la izquierda)
    draw_color_selector()

    # Instrucciones
    txt = SMALL_FONT.render("Haz click en los sliders para ajustar el rango HSV. Haz click en el color para seleccionarlo.", True, (0, 0, 0))
    screen.blit(txt, (60, 20))
    txt2 = SMALL_FONT.render("Pulsa ESC o cierra la ventana para salir.", True, (0, 0, 0))
    screen.blit(txt2, (60, 40))

    # Mostrar frame original y máscara (más centrados)
    surf_frame = cvimg_to_pygame(frame_small)
    surf_mask = mask_to_pygame(mask)
    screen.blit(surf_frame, (500, 60))
    screen.blit(surf_mask, (850, 60))
    pygame.draw.rect(screen, (0, 0, 0), (500, 60, 340, 260), 2)
    pygame.draw.rect(screen, (0, 0, 0), (850, 60, 340, 260), 2)
    screen.blit(SMALL_FONT.render("Original", True, (0,0,0)), (500, 40))
    screen.blit(SMALL_FONT.render("Máscara", True, (0,0,0)), (850, 40))

    pygame.display.flip()
    clock.tick(30)