import cv2
import mediapipe as mp
import pyautogui

# Inicializa o Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Tamanho da tela
screen_width, screen_height = pyautogui.size()

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Estado do clique e arraste
is_dragging = False
previous_x, previous_y = pyautogui.position()
sensitivity = 3  # Reduz a sensibilidade do cursor
hand_visible = False
windows_tab_triggered = False  # Evita acionamento contínuo do Windows + Tab

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_visible = True  # A mão está visível
        for hand_landmarks in result.multi_hand_landmarks:
            # Calcula o centro da mão (média de todas as landmarks)
            cx = int(sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark) * screen_width)
            cy = int(sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark) * screen_height)

            # Suaviza o movimento do cursor
            smoothed_x = previous_x + (cx - previous_x) // sensitivity
            smoothed_y = previous_y + (cy - previous_y) // sensitivity
            pyautogui.moveTo(smoothed_x, smoothed_y)
            previous_x, previous_y = smoothed_x, smoothed_y

            # Calcula quais dedos estão esticados
            def is_finger_extended(finger_tip, finger_dip):
                return finger_tip.y < finger_dip.y  # Dedos esticados têm a ponta acima da articulação DIP

            # Verifica os dedos indicador, médio, anelar e mínimo
            four_fingers_extended = all([
                is_finger_extended(
                    hand_landmarks.landmark[getattr(mp_hands.HandLandmark, f"{finger}_TIP")],
                    hand_landmarks.landmark[getattr(mp_hands.HandLandmark, f"{finger}_DIP")]
                )
                for finger in ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
            ])

            # Aciona Windows + Tab ao abrir os quatro dedos
            if four_fingers_extended:
                if not windows_tab_triggered:
                    pyautogui.hotkey('win', 'tab')
                    windows_tab_triggered = True
            else:
                windows_tab_triggered = False  # Permite disparar novamente

            # Desenha landmarks e conexões da mão
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        hand_visible = False  # A mão saiu do campo de visão

    # Ajusta a posição inicial do cursor ao retornar a mão ao campo de visão
    if not hand_visible:
        previous_x, previous_y = pyautogui.position()

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
