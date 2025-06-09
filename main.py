import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

from pandas.core import frame

print(sys.executable)
mpl.use('TkAgg')
def point_px_to_m(px_value,scale):
    return px_value * scale

def calculate_scale_px_to_m_slide(video_path,h_m_slide):
    ret,frame_video = cv2.VideoCapture(video_path).read()
    draw = False

    if not ret:
        print("Error al leer el frame")
        exit()  # o maneja el error
    points = []
    def click_event(event, x, y, flags, param):
        nonlocal points,draw

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points)< 2:
                points.append((x,y))
                cv2.circle(frame_video, (x, y), 2, (255, 0, 0), -1)
                cv2.imshow('Selecciona extremos', frame_video)
                print(f"Punto {len(points)} seleccionado: ({x}, {y})")
                if len(points) == 1:
                    draw = True
            else:
                print("Ya seleccionaste dos puntos.")
                draw = False
        elif event == cv2.EVENT_MOUSEMOVE and draw:
            frame_copy = frame_video.copy()
            cv2.circle(frame_copy, points[0], 5, (0, 0, 255), -1)

            # Dibuja la línea desde el primer punto al mouse
            cv2.line(frame_copy, points[0], (x, y), (255, 0, 0), 2)
            cv2.imshow("Selecciona extremos", frame_copy)

    frame_copy = frame_video.copy()
    cv2.imshow("Selecciona extremos", frame_copy)
    cv2.setMouseCallback("Selecciona extremos", click_event)
    print("Haz click en el punto más alto y luego en el más bajo del tobogán.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        raise ValueError("No se seleccionaron exactamente dos puntos.")

    h_px_slide = abs(points[1][1] - points[0][1])
    scale_m_px = h_m_slide / h_px_slide
    print(f"Escala metros/píxel: {scale_m_px:.5f}")
    return scale_m_px



def trackear_video(video_path, csv_path,px_to_m):

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Error capturing video")
        return [],[],[]
    #selecciono altura
    height = frame.shape[0]
    #Calculo de altura en pixeles del tobogan

    #Seleccion Manual ROI
    bbox = cv2.selectROI("Selecciona el cuerpo",frame,fromCenter=False,showCrosshair=False)
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    times = []
    xs = []
    ys = []
    fps = cap.get(cv2.CAP_PROP_FPS) # Obtener FPS del video
    frame_cont = 0 # Contador de frames

    with open(csv_path,'w',newline= '') as movement_file:
        writer = csv.writer(movement_file)
        writer.writerow(["time(s)","x","y"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = bbox
                cx = int(x + w / 2)
                cy = int(y + w / 2)
                t = frame_cont / fps

                writer.writerow([t, cx, cy])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)


                times.append(t)
                cx_m = point_px_to_m(cx,px_to_m)
                xs.append(cx_m)
                cy_cart = height - y
                cy_m = point_px_to_m(cy_cart,px_to_m)
                ys.append(cy_m)
                p1 = (int(x), int(y))
                p2 = (int(x + w), int(y + h))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Tracking perdido", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_cont += 1

    cap.release()
    cv2.destroyAllWindows()
    return times, xs, ys

def graficar_analizar(times,xs,ys):
    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)

    #Creio dataframe para guardar listas de x,y,t
    df = pd.DataFrame({
        'Tiempo (s)': times,
        'Posición X (m)': xs,
        'Posición Y (m)': ys
    })
    df.to_csv('positions.csv', index=False)

    # Mostrar el DataFrame en una ventana emergente usando tkinter--------
    import tkinter as tk
    from pandastable import Table
    root = tk.Tk()
    root.title('Datos de Tracking (positions.csv)')
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)
    pt = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
    pt.show()
    root.mainloop()
    #----------------------------------------------------------------------
    # Suavizado Whittaker-Eilers para xs y ys  
    from whittaker_eilers import WhittakerSmoother
    smoother = WhittakerSmoother(lmbda=10**2, order=2, data_length=len(times))  # El parámetro de suavizado es el primer argumento (ajusta si quieres más/menos suavizado)
    xs_smooth = smoother.smooth(xs)
    ys_smooth = smoother.smooth(ys)
    
    # Cálculo de velocidad y aceleración 
    vxs = np.gradient(xs_smooth, times)
    vys = np.gradient(ys_smooth, times)
    axs = np.gradient(vxs, times)
    ays = np.gradient(vys, times)

    fig, axs_plot = plt.subplots(3,1,figsize=(10,12))

    # Graficar posiciones
    axs_plot[0].plot(times, xs_smooth, label='X ', color='blue')
    axs_plot[0].plot(times, ys_smooth, label='Y', color='green')
    axs_plot[0].set_title('Posición vs Tiempo')
    axs_plot[0].set_ylabel('Posición (m)')
    axs_plot[0].legend()

    # Graficar velocidades
    axs_plot[1].plot(times, vxs, label='Vx', color='pink')
    axs_plot[1].plot(times, vys, label='Vy', color='purple')
    axs_plot[1].set_title('Velocidad vs Tiempo')
    axs_plot[1].set_ylabel('Velocidad (m/s)')
    axs_plot[1].legend()

    # Graficar aceleraciones
    axs_plot[2].plot(times, axs, label='Ax', color='red')
    axs_plot[2].plot(times, ays, label='Ay', color='orange')
    axs_plot[2].set_title('Aceleración vs Tiempo')
    axs_plot[2].set_xlabel('Tiempo (s)')
    axs_plot[2].set_ylabel('Aceleración (m/s²)')
    axs_plot[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    video_path = "videos/i1.mp4"
    h_m_slide = 2.94
    px_to_m = calculate_scale_px_to_m_slide(video_path,h_m_slide)
    times, xs, ys = trackear_video(video_path, "positions.csv",px_to_m)
    if times:  # Solo graficamos si hubo datos
        graficar_analizar(times, xs, ys)


if __name__ == "__main__":
    main()

