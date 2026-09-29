/*
 * Tap-Rhythm Classifier - TensorFlow Lite Micro on FRDM-MCXN236.
 *
 * Tap SW2 six times. The five gaps between taps are turned into features,
 * a small int8 neural network classifies the rhythm and the RGB LED shows it:
 *   steady -> green, gallop -> blue, speed_up -> red, slow_down -> magenta,
 *   irregular (none of the above) -> white
 * SW3 cancels a half-finished sequence.
 *
 * Log mode (hold SW3 while resetting the board): SW3 picks the label you are
 * about to tap (LED shows it) and every sequence is printed as a CSV row
 * "label,i1,i2,i3,i4,i5" that train/train.py --real can learn from.
 */

#include <math.h>
#include <stdio.h>

#include <zephyr/kernel.h>
#include <zephyr/drivers/gpio.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

#define NUM_TAPS       (NUM_INTERVALS + 1)
#define DEBOUNCE_MS    40
#define TAP_TIMEOUT_MS 3000

namespace {

/* ---------------------------------------------------------------- GPIO -- */

const struct gpio_dt_spec led_red = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
const struct gpio_dt_spec led_green = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);
const struct gpio_dt_spec led_blue = GPIO_DT_SPEC_GET(DT_ALIAS(led2), gpios);
const struct gpio_dt_spec btn_tap = GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);
const struct gpio_dt_spec btn_cancel = GPIO_DT_SPEC_GET(DT_ALIAS(sw1), gpios);

struct gpio_callback tap_cb;
struct gpio_callback cancel_cb;

/* LED colour per class, as {red, green, blue}. */
const uint8_t class_colors[NUM_CLASSES][3] = {
	{0, 1, 0}, /* steady    - green   */
	{0, 0, 1}, /* gallop    - blue    */
	{1, 0, 0}, /* speed_up  - red     */
	{1, 0, 1}, /* slow_down - magenta */
	{1, 1, 1}, /* irregular - white   */
};

/* Tap timestamps, written from the button ISR. */
int64_t tap_ms[NUM_TAPS];
volatile int tap_count;
volatile bool cancel_requested;
K_SEM_DEFINE(event_sem, 0, 1);

/* ---------------------------------------------------------------- TFLM -- */

constexpr int kArenaSize = 4 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;

void set_led(const uint8_t rgb[3])
{
	gpio_pin_set_dt(&led_red, rgb[0]);
	gpio_pin_set_dt(&led_green, rgb[1]);
	gpio_pin_set_dt(&led_blue, rgb[2]);
}

void leds_off(void)
{
	static const uint8_t off[3] = {0, 0, 0};

	set_led(off);
}

void tap_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
	int64_t now = k_uptime_get();

	/* Ignore contact bounce right after the previous tap. */
	if (tap_count > 0 && now - tap_ms[tap_count - 1] < DEBOUNCE_MS) {
		return;
	}
	if (tap_count < NUM_TAPS) {
		tap_ms[tap_count++] = now;
		k_sem_give(&event_sem);
	}
}

void cancel_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
	static int64_t last_ms;
	int64_t now = k_uptime_get();

	/* One press of SW3 bounces many times; accept one edge per 250 ms. */
	if (now - last_ms < 250) {
		return;
	}
	last_ms = now;
	cancel_requested = true;
	k_sem_give(&event_sem);
}

int init_button(const struct gpio_dt_spec *btn, struct gpio_callback *cb,
		gpio_callback_handler_t handler)
{
	if (!gpio_is_ready_dt(btn) ||
	    gpio_pin_configure_dt(btn, GPIO_INPUT) != 0 ||
	    gpio_pin_interrupt_configure_dt(btn, GPIO_INT_EDGE_TO_ACTIVE) != 0) {
		return -1;
	}
	gpio_init_callback(cb, handler, BIT(btn->pin));
	return gpio_add_callback(btn->port, cb);
}

int init_gpio(void)
{
	const struct gpio_dt_spec *leds[] = {&led_red, &led_green, &led_blue};

	for (const struct gpio_dt_spec *led : leds) {
		if (!gpio_is_ready_dt(led) ||
		    gpio_pin_configure_dt(led, GPIO_OUTPUT_INACTIVE) != 0) {
			return -1;
		}
	}
	if (init_button(&btn_tap, &tap_cb, tap_isr) != 0 ||
	    init_button(&btn_cancel, &cancel_cb, cancel_isr) != 0) {
		return -1;
	}
	return 0;
}

int init_model(void)
{
	const tflite::Model *model = tflite::GetModel(g_tap_model);

	if (model->version() != TFLITE_SCHEMA_VERSION) {
		printf("Model schema %lu != supported %d\n",
		       (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
		return -1;
	}

	/* Register only the ops this model uses (see train.py output). */
	static tflite::MicroMutableOpResolver<2> resolver;
	resolver.AddFullyConnected();
	resolver.AddSoftmax();

	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
							   kArenaSize);
	interpreter = &static_interpreter;
	if (interpreter->AllocateTensors() != kTfLiteOk) {
		printf("AllocateTensors() failed\n");
		return -1;
	}
	input = interpreter->input(0);
	output = interpreter->output(0);

	printf("Model: %u bytes, arena used %u of %d bytes\n", g_tap_model_len,
	       (unsigned)interpreter->arena_used_bytes(), kArenaSize);
	printf("Input  quantization: scale=%f zero_point=%d\n",
	       (double)input->params.scale, (int)input->params.zero_point);
	printf("Output quantization: scale=%f zero_point=%d\n",
	       (double)output->params.scale, (int)output->params.zero_point);
	return 0;
}

/*
 * Run the network on one float feature vector.
 * Returns the winning class and fills probs[] with dequantized scores.
 */
int classify(const float features[NUM_INTERVALS], float probs[NUM_CLASSES],
	     uint32_t *cycles)
{
	/* Quantize: q = round(x / scale) + zero_point, clamped to int8. */
	for (int i = 0; i < NUM_INTERVALS; i++) {
		int32_t q = (int32_t)lroundf(features[i] / input->params.scale) +
			    input->params.zero_point;
		input->data.int8[i] = (int8_t)CLAMP(q, -128, 127);
	}

	uint32_t start = k_cycle_get_32();
	TfLiteStatus status = interpreter->Invoke();
	*cycles = k_cycle_get_32() - start;
	if (status != kTfLiteOk) {
		printf("Invoke failed\n");
		return -1;
	}

	/* Dequantize: x = (q - zero_point) * scale. */
	int best = 0;
	for (int c = 0; c < NUM_CLASSES; c++) {
		probs[c] = (output->data.int8[c] - output->params.zero_point) *
			   output->params.scale;
		if (probs[c] > probs[best]) {
			best = c;
		}
	}
	return best;
}

void print_result(int cls, const float probs[NUM_CLASSES], uint32_t cycles)
{
	printf("  -> %-9s (", g_class_names[cls]);
	for (int c = 0; c < NUM_CLASSES; c++) {
		printf("%s%s %.2f", c ? ", " : "", g_class_names[c], (double)probs[c]);
	}
	printf(")  inference %u us\n", (unsigned)k_cyc_to_us_floor32(cycles));
}

/* Classify the canned vectors from train.py (one per class) so the model can be checked
 * on the device without anyone touching the buttons.
 */
void self_test(void)
{
	int passed = 0;

	printf("\nSelf-test:\n");
	for (int t = 0; t < NUM_TESTS; t++) {
		float probs[NUM_CLASSES];
		uint32_t cycles;
		int cls = classify(g_test_features[t], probs, &cycles);
		bool ok = cls == g_test_labels[t];

		passed += ok;
		printf("  [%s] expected %-9s", ok ? "PASS" : "FAIL",
		       g_class_names[g_test_labels[t]]);
		print_result(cls, probs, cycles);
	}
	printf("Self-test: %d/%d PASS\n\n", passed, NUM_TESTS);
}

void reset_taps(void)
{
	unsigned int key = irq_lock();

	tap_count = 0;
	irq_unlock(key);
}

} /* namespace */

int main(void)
{
	printf("\n=== Tap-Rhythm Classifier (TFLite Micro on %s) ===\n", CONFIG_BOARD);

	if (init_gpio() != 0 || init_model() != 0) {
		printf("Init failed\n");
		return 0;
	}
	self_test();

	/* Holding SW3 during boot selects log mode for collecting real data. */
	bool log_mode = gpio_pin_get_dt(&btn_cancel) == 1;
	int log_label = 0;

	if (log_mode) {
		printf("LOG MODE: SW3 selects label, SW2 x%d records a CSV row\n", NUM_TAPS);
		printf("Label: %d (%s)\n", log_label, g_class_names[log_label]);
		set_led(class_colors[log_label]);
	} else {
		printf("Tap SW2 %d times in a rhythm: steady / gallop / speed_up / "
		       "slow_down (anything else -> irregular). SW3 cancels.\n", NUM_TAPS);
	}

	int last_count = 0;

	while (true) {
		k_sem_take(&event_sem, K_MSEC(200));

		if (cancel_requested) {
			cancel_requested = false;
			reset_taps();
			last_count = 0;
			if (log_mode) {
				log_label = (log_label + 1) % NUM_CLASSES;
				printf("Label: %d (%s)\n", log_label, g_class_names[log_label]);
				set_led(class_colors[log_label]);
			} else {
				printf("Cancelled\n");
				leds_off();
			}
			continue;
		}

		int count = tap_count;

		if (count > 0 && count < NUM_TAPS &&
		    k_uptime_get() - tap_ms[count - 1] > TAP_TIMEOUT_MS) {
			printf("Timeout after %d taps, start again\n", count);
			reset_taps();
			last_count = 0;
			continue;
		}
		if (count != last_count && count < NUM_TAPS) {
			if (count == 1 && !log_mode) {
				leds_off();
			}
			printf("tap %d\n", count);
			last_count = count;
		}
		if (count < NUM_TAPS) {
			continue;
		}

		/* Six taps recorded: build tempo-invariant features. */
		float gaps[NUM_INTERVALS];
		float mean = 0.0f;

		for (int i = 0; i < NUM_INTERVALS; i++) {
			gaps[i] = (float)(tap_ms[i + 1] - tap_ms[i]);
			mean += gaps[i] / NUM_INTERVALS;
		}

		float features[NUM_INTERVALS];

		for (int i = 0; i < NUM_INTERVALS; i++) {
			features[i] = gaps[i] / mean;
		}

		printf("gaps ms: %d %d %d %d %d\n", (int)gaps[0], (int)gaps[1],
		       (int)gaps[2], (int)gaps[3], (int)gaps[4]);
		if (log_mode) {
			printf("CSV,%d,%d,%d,%d,%d,%d\n", log_label, (int)gaps[0],
			       (int)gaps[1], (int)gaps[2], (int)gaps[3], (int)gaps[4]);
		}

		float probs[NUM_CLASSES];
		uint32_t cycles;
		int cls = classify(features, probs, &cycles);

		if (cls >= 0) {
			print_result(cls, probs, cycles);
			if (!log_mode) {
				set_led(class_colors[cls]);
			}
		}
		reset_taps();
		last_count = 0;
	}
	return 0;
}
