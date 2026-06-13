; Traverse a linked list, sum the values
;
; Each node is 8 bytes: [value: u32, next_ptr: u32]
; Demonstrates cache replacement policy effects on pointer-chasing
; workloads with zero spatial locality.
;
; Config provides:
;   a0 = head pointer (address of first node)

    mv   t0, a0
    li   a0, 0

loop:
    beqz t0, done      ; if ptr == null, done
    lw   t1, 0(t0)     ; t1 = node->value
    add  a0, a0, t1    ; sum += value
    lw   t0, 4(t0)     ; ptr = node->next
    j    loop

done:
    ebreak
