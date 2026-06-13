; Bubble sort an array of u32 values
;
; Dense branch-heavy code with data-dependent branches.
; The comparison outcomes are unpredictable, depending on data order.
;
; Config provides:
;   a0 = array length
;   a1 = array address

    mv   t6, a0         ; n = length

outer:
    addi t6, t6, -1     ; n-- (last pass only needs n-1 comparisons)
    beqz t6, done       ; if n == 0, fully sorted
    li   t0, 0          ; i = 0
    li   t5, 0          ; swapped = false

inner:
    beq  t0, t6, check  ; if i == n, end inner loop

    ; Load array[i] and array[i+1]
    slli t1, t0, 2      ; byte offset for i
    add  t1, t1, a1     ; &array[i]
    lw   t2, 0(t1)      ; array[i]
    lw   t3, 4(t1)      ; array[i+1]

    ; Compare: if array[i] <= array[i+1], no swap needed
    bge  t3, t2, no_swap

    ; Swap
    sw   t3, 0(t1)
    sw   t2, 4(t1)
    li   t5, 1          ; swapped = true

no_swap:
    addi t0, t0, 1      ; i++
    j    inner

check:
    bnez t5, outer      ; if swapped, do another pass

done:
    lw   a0, 0(a1)      ; return sorted[0]
    ebreak
