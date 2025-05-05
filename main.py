import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
def trackear_video(video_path, csv_path = "deslizamiento.csv"):

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        print("Error capturing video")
        return [],[],[]
    #selecciono altura
    height = frame.shape[0]
    #Seleccion Manual ROI
    bbox = cv2.selectROI("Selecciona el cuerpo",frame,fromCenter=False,showCrosshair=False)
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    times = []
    xs = []
    ys= []
    start_time = time.time()

    with open(csv_path,'w',newline= '') as movement_file:
        writer = csv.writer(movement_file)
        writer.writerow(["time(s)","x","y"])

        while  True:
            ret, frame = cap.read()
            if not ret:
                break
            success, bbox = tracker.update(frame)
            if success:
                x,y,w,h = bbox
                cx = int(x + w/ 2)
                cy = int(y + w/ 2)
                t = time.time() - start_time

                writer.writerow([t,cx,cy,w,h])
                cv2.circle(frame,(cx,cy),5,(0,255,0),-1)

                times.append(t)
                xs.append(cx)
                cy = height - y
                ys.append(cy)
                p1 = (int(x), int(y))
                p2 = (int(x + w), int(y + h))
                cv2.rectangle(frame,p1,p2,(255,0,0),2)

            else:
                cv2.putText(frame, "Tracking perdido", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return times, xs, ys

def graficar(times,xs,ys):
    times = np.array(times)
    xs = np.array(xs)
    ys = np.array(ys)

    vxs = np.gradient(xs,times)
    vys = np.gradient(ys,times)
    axs = np.gradient(vxs,times)
    ays = np.gradient(vys,times)

    fig, axs_plot = plt.subplots(3,1,figsize=(10,12))

    axs_plot[0].plot(times, xs, label='X')
    axs_plot[0].plot(times, ys, label='Y')
    axs_plot[0].set_title('Posición vs Tiempo')
    axs_plot[0].set_ylabel('Posición (px)')
    axs_plot[0].legend()

    axs_plot[1].plot(times, vxs, label='Vx')
    axs_plot[1].plot(times, vys, label='Vy')
    axs_plot[1].set_title('Velocidad vs Tiempo')
    axs_plot[1].set_ylabel('Velocidad (px/s)')
    axs_plot[1].legend()

    axs_plot[2].plot(times, axs, label='Ax')
    axs_plot[2].plot(times, ays, label='Ay')
    axs_plot[2].set_title('Aceleración vs Tiempo')
    axs_plot[2].set_xlabel('Tiempo (s)')
    axs_plot[2].set_ylabel('Aceleración (px/s²)')
    axs_plot[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    times, xs, ys = trackear_video("videos/i1.mp4", "deslizamiento.csv")
    if times:  # Solo graficamos si hubo datos
        graficar(times, xs, ys)


if __name__ == "__main__":
    main()