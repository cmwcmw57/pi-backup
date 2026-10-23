/*
 * Stage 0 -- toolchain sanity check only. No firmware/common logic
 * touched yet; this just proves Arduino IDE 2 + STM32duino can
 * compile and flash the Nucleo-F446RE over its onboard ST-Link.
 *
 * LED_BUILTIN maps to LD2 (the onboard green LED, physically on pin
 * PA5) on all Nucleo-64 boards including the F446RE -- using the
 * built-in macro instead of hardcoding PA5 so this stays portable if
 * the board selection ever changes.
 */
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(500);
  digitalWrite(LED_BUILTIN, LOW);
  delay(500);
}
