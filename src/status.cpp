#include "xnor_nn.h"

void xnor_nn_get_status_message(char *msg, xnor_nn_status_t status) {
    (void)msg;
    (void)status;
    for (int i = 0;; i++){
        switch (status) {
            case xnor_nn_success: msg[i] = "success"[i]; break;
            case xnor_nn_error_memory: msg[i] = "memory_error"[i]; break;
            case xnor_nn_error_invalid_input: msg[i] = "bad_input"[i]; break;
        }
        if (msg[i] == '\0') break;
    }
}
