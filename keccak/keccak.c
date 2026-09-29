#include <stdio.h>
#include <stdint.h>

#define LANE(x,y) ((x) + 5 * (y))

typedef enum {
    SHA3_224,
    SHA3_256,
    SHA3_384,
    SHA3_512,
    SHAKE128,
    SHAKE256,
    KECCAK_MODE_COUNT
} keccak_mode_e;    

typedef struct {
    size_t rate;
    uint8_t domain;
} keccak_params;

static const keccak_params PARAMS[KECCAK_MODE_COUNT] = {
    [SHA3_224] = {144, 0x06},    // designated initializers: index by enum name
    [SHA3_256] = {136, 0x06},
    [SHA3_384] = {104, 0x06},
    [SHA3_512] = { 72, 0x06},
    [SHAKE128] = {168, 0x1F},
    [SHAKE256] = {136, 0x1F},
};


typedef uint64_t keccak_state[25];

void absorb(keccak_state st,  keccak_mode_e mode, uint8_t* msg, size_t len){
    
    size_t absorb_limit = PARAMS[mode].rate;
    size_t i;
    for (i = 0; (i < len) && (i < absorb_limit); i++){
        st[i/8]^= ((uint64_t)msg[i] << ((i*8) % 64));
    }
    st[i / 8] ^= (uint64_t)PARAMS[mode].domain << (8 * (i % 8));
    st[(absorb_limit - 1) / 8] ^= (uint64_t)0x80 << (8 * ((absorb_limit - 1) % 8));

}

void print(keccak_state st){
    for (int i = 0; i < 25; i++)
    printf("lane %2d: %016llx\n", i, (unsigned long long)st[i]);
};


int main (){

    keccak_state state = {0};    

    char msg[] = {"mensagemginhainicial\n"};

    absorb(state, SHA3_512, msg, sizeof(msg));
 
    print(state);

    return 0;
}
