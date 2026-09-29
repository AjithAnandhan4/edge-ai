/*
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "output_handler.hpp"

#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>

/* Green LED (alias led1 on FRDM-MCXN236). */
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);

/* Each prediction is shown for this long, with software PWM at 1 kHz. */
#define SHOW_MS    50
#define PWM_PERIOD_US 1000

void HandleOutputInit(void)
{
	if (!gpio_is_ready_dt(&led) ||
	    gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE) != 0) {
		MicroPrintf("LED init failed");
	}
}

void HandleOutput(float x_value, float y_value, uint32_t invoke_cycles)
{
	/* Log the current X and Y values */
	MicroPrintf("x_value: %f, y_value: %f, invoke: %u ns",
		    static_cast < double > (x_value),
		    static_cast < double > (y_value),
		    (unsigned)k_cyc_to_ns_floor64(invoke_cycles));

	/* Map y in [-1, 1] to a duty cycle in [0, 100] percent. */
	int duty = (int)((y_value + 1.0f) * 50.0f);
	duty = duty < 0 ? 0 : (duty > 100 ? 100 : duty);
	int on_us = PWM_PERIOD_US * duty / 100;

	for (int t = 0; t < SHOW_MS * 1000; t += PWM_PERIOD_US) {
		if (on_us > 0) {
			gpio_pin_set_dt(&led, 1);
			k_busy_wait(on_us);
		}
		if (on_us < PWM_PERIOD_US) {
			gpio_pin_set_dt(&led, 0);
			k_busy_wait(PWM_PERIOD_US - on_us);
		}
	}
}
