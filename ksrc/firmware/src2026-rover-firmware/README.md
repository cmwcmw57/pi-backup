# STM32 Rover Motor Control

This project is an STM32-based motor control firmware for a 4-wheel rover. It features PWM DC motor control, quadrature encoder reading, basic kinematics calculation, and **Feetech STS3215 Serial Bus Servo control** for steering.

## Hardware Specifications
- **Microcontroller**: STM32F446RET6 (NUCLEO-F446RE)
- **Motor Driver**: Cytron MDD3A (4-channel, 2-PWM mode)
- **Drive Motors**: 4 x SPG30E-GR131 (12V DC Geared Motor)
  - Gear Ratio: 131:1 / Ticks per Revolution: 6812.0
- **Steering Servos**: 4 x Feetech STS3215 (Serial Bus Servo)
  - ID 1~4, Daisy-chained
  - 1Mbps Single-Wire Half-Duplex Communication

## Features
- **Drive Motor Control**: Forward, reverse, and stop using hardware PWM.
- **Encoder Reading**: Hardware timer-based x4 quadrature decoding.
- **Servo Motor Control (NEW)**:
  - Custom UART protocol implementation without HAL overhead.
  - **Single-Wire Half-Duplex (HDSEL) mode**: Direct connection without external diode/adapter board!
  - Sync-write multiple servos simultaneously.
- **Interactive Terminal Interface**: Control servo angles in real-time via Serial Monitor.

## Terminal Commands (115200 bps, LF or CRLF)
| Command | Description |
| :--- | :--- |
| `PING` | Check connection status for all 4 servos |
| `T <a1> <a2> <a3> <a4>` | Move servos to target angles (e.g., `T 45 -45 45 -45`) |
| `P` | Print current angles of all servos |
| `ON` / `OFF` | Enable or disable servo torque |

## Pin Configuration (Wiring Guide)

### 1. Drive Motors (Cytron MDD3A PWM)
| STM32 Pin | Timer/Channel | 연결 대상 (Target) |
| :--- | :--- | :--- |
| **PA5** | `TIM2_CH1` | MDD3A #1 → M1A |
| **PB3** | `TIM2_CH2` | MDD3A #1 → M1B |
| **PB14**| `TIM12_CH1` | MDD3A #1 → M2A |
| **PB15**| `TIM12_CH2` | MDD3A #1 → M2B |
| **PA6** | `TIM3_CH1` | MDD3A #2 → M1A |
| **PA7** | `TIM3_CH2` | MDD3A #2 → M1B |
| **PB0** | `TIM3_CH3` | MDD3A #2 → M2A |
| **PB1** | `TIM3_CH4` | MDD3A #2 → M2B |

### 2. Encoders (모터의 노랑/흰색 선)
| STM32 Pin | Timer/Channel | 연결 대상 (Target) |
| :--- | :--- | :--- |
| **PA8** | `TIM1_CH1` | Motor 1 인코더 Yellow (채널A) |
| **PA9** | `TIM1_CH2` | Motor 1 인코더 White (채널B) |
| **PB6** | `TIM4_CH1` | Motor 2 인코더 Yellow |
| **PB7** | `TIM4_CH2` | Motor 2 인코더 White |
| **PA0** | `TIM5_CH1` | Motor 3 인코더 Yellow |
| **PA1** | `TIM5_CH2` | Motor 3 인코더 White |
| **PC6** | `TIM8_CH1` | Motor 4 인코더 Yellow |
| **PC7** | `TIM8_CH2` | Motor 4 인코더 White |

### 3. Steering Servos (Feetech STS3215)
> ⚠️ **주의 (핀 변경 내역)**: 기존에 논의되었던 UART1(PA9/PA10)은 Motor 1 엔코더(PA9)와의 **핀 충돌을 피하기 위해 USART3(PB10/PB11)로 교체**되었습니다.
> **Direct Single-Wire Connection** (No external adapter needed)

| STM32 Pin | 연결 대상 (Target) | Note |
| :--- | :--- | :--- |
| **PB10** (`USART3_TX`) | Servo **BUS (Data)** | Single-Wire HDSEL mode (Open-Drain). Connect directly to servo signal wire. |

### 4. 공통 전원 및 GND (Power/GND)
| STM32 Pin | 연결 대상 (Target) | Note |
| :--- | :--- | :--- |
| **GND** | MDD3A #1 GND, MDD3A #2 GND, 모터 4개 인코더 Green(각각), 외부 12V 전원 GND, 서보 모터 GND | **MUST be connected to Common Ground!** |
| **3.3V 또는 5V** | 모터 4개 인코더 Blue(각각) | 전압 레벨 확인 필요 |
| **Not Connected** | 서보 모터 VCC | ⚠️ **STM32에 절대 연결 금지!** 12V 외부 전원에만 직결 |

### 5. Debug & Terminal Interface
| STM32 Pin | 연결 대상 (Target) | Note |
| :--- | :--- | :--- |
| **PA2** | `USART2_TX` | ST-LINK VCP (115200 bps) |
| **PA3** | `USART2_RX` | ST-LINK VCP (115200 bps, Terminal Command Input) |

## How to Build
This project uses a standard Makefile. 
```bash
make clean
make -j4
```
Flash the resulting `build/rover_motor.elf` or `.hex` file using STM32CubeIDE or STM32CubeProgrammer.
