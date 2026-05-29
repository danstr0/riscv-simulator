; Two cores atomically incrementing a shared counter
;
; Each core increments the counter 10 times using LR/SC. If SC fails
; (another core wrote between LR and SC), the core retries.
;
; Config provides:
;   a0 = core ID (0 or 1, set per-core)
;   a1 = address of shared counter
;   a2 = number of increments per core
;
; Returns: a0 = final counter value (2 * a2 = 20)

    mv   t0, a2

loop:
    beqz t0, done

retry:
    lr.w t1, (a1)       ; t1 = *counter (load-reserved)
    addi t2, t1, 1      ; t2 = counter + 1
    sc.w t3, t2, (a1)   ; try store; t3 = 0 on success
    bnez t3, retry      ; if SC failed, retry

    addi t0, t0, -1     ; remaining--
    j    loop

done:
    lw   a0, 0(a1)
    ebreak
