; 4x4 matrix multiplication (C = A × B)
;
; Demonstrates cache spatial/temporal locality tradeoffs.
; Row-major layout -> accessing B's columns causes cache misses.
;
; Config provides:
;   a0 = N (matrix dimension: 4)
;   a1 = address of A
;   a2 = address of B
;   a3 = address of C (output)

    ; Outer loop: i = 0..N-1
    li   t0, 0         ; i = 0
outer:
    beq  t0, a0, done

    ; Middle loop: j = 0..N-1
    li   t1, 0         ; j = 0
middle:
    beq  t1, a0, next_i

    ; Inner loop: sum = 0; for k = 0..N-1: sum += A[i][k] * B[k][j]
    li   t6, 0         ; sum = 0
    li   t2, 0         ; k = 0
inner:
    beq  t2, a0, store

    ; t3 = &A[i][k] = a1 + (i*N + k)*4
    mul  t3, t0, a0    ; i*N
    add  t3, t3, t2    ; i*N + k
    slli t3, t3, 2     ; byte offset
    add  t3, t3, a1    ; address
    lw   t4, 0(t3)     ; A[i][k]

    ; t3 = &B[k][j] = a1 + (k*N + j)*4
    mul  t3, t2, a0    ; k*N
    add  t3, t3, t1    ; k*N + j
    slli t3, t3, 2
    add  t3, t3, a2
    lw   t5, 0(t3)     ; B[k][j]

    mul  t4, t4, t5    ; A[i][k] * B[k][j]
    add  t6, t6, t4    ; sum += product

    addi t2, t2, 1     ; k++
    j    inner

store:
    ; C[i][j] = sum -> a3 + (i*N + j)*4 
    mul  t3, t0, a0
    add  t3, t3, t1
    slli t3, t3, 2
    add  t3, t3, a3
    sw   t6, 0(t3)

    addi t1, t1, 1     ; j++
    j    middle

next_i:
    addi t0, t0, 1     ; i++
    j    outer

done:
    lw   a0, 0(a3)     ; return C[0][0]
    ebreak
