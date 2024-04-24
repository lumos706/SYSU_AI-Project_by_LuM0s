(define (problem PPT1)
  (:domain puzzle)
  (:objects 
    num1 num2 num3 num4 num5 num6 num7 num8 num9 num10 num11 num12 num13 num14 num15 - num
    loc-1-1 loc-1-2 loc-1-3 loc-1-4
    loc-2-1 loc-2-2 loc-2-3 loc-2-4
    loc-3-1 loc-3-2 loc-3-3 loc-3-4
    loc-4-1 loc-4-2 loc-4-3 loc-4-4 - loc
  )
; 6 10 3 15
; 14 8 7 11
; 5 1 0 2
; 13 12 9 4
  (:init 
    (at num6 loc-1-1) (at num10 loc-1-2) (at num3 loc-1-3) (at num15 loc-1-4)
    (at num14 loc-2-1) (at num8 loc-2-2) (at num7 loc-2-3) (at num11 loc-2-4)
    (at num5 loc-3-1) (at num1 loc-3-2) (at B loc-3-3) (at num2 loc-3-4)
    (at num13 loc-4-1) (at num12 loc-4-2) (at num9 loc-4-3) (at num4 loc-4-4)
    ; Add adjacency relations here
    (adjacent loc-1-1 loc-1-2) (adjacent loc-1-1 loc-2-1)
    (adjacent loc-1-2 loc-1-1) (adjacent loc-1-2 loc-1-3) (adjacent loc-1-2 loc-2-2)
    (adjacent loc-1-3 loc-1-2) (adjacent loc-1-3 loc-1-4) (adjacent loc-1-3 loc-2-3)
    (adjacent loc-1-4 loc-1-3) (adjacent loc-1-4 loc-2-4)
    (adjacent loc-2-1 loc-1-1) (adjacent loc-2-1 loc-2-2) (adjacent loc-2-1 loc-3-1)
    (adjacent loc-2-2 loc-1-2) (adjacent loc-2-2 loc-2-1) (adjacent loc-2-2 loc-2-3) (adjacent loc-2-2 loc-3-2)
    (adjacent loc-2-3 loc-1-3) (adjacent loc-2-3 loc-2-2) (adjacent loc-2-3 loc-2-4) (adjacent loc-2-3 loc-3-3)
    (adjacent loc-2-4 loc-1-4) (adjacent loc-2-4 loc-2-3) (adjacent loc-2-4 loc-3-4)
    (adjacent loc-3-1 loc-2-1) (adjacent loc-3-1 loc-3-2) (adjacent loc-3-1 loc-4-1)
    (adjacent loc-3-2 loc-2-2) (adjacent loc-3-2 loc-3-1) (adjacent loc-3-2 loc-3-3) (adjacent loc-3-2 loc-4-2)
    (adjacent loc-3-3 loc-2-3) (adjacent loc-3-3 loc-3-2) (adjacent loc-3-3 loc-3-4) (adjacent loc-3-3 loc-4-3)
    (adjacent loc-3-4 loc-2-4) (adjacent loc-3-4 loc-3-3) (adjacent loc-3-4 loc-4-4)
    (adjacent loc-4-1 loc-3-1) (adjacent loc-4-1 loc-4-2)
    (adjacent loc-4-2 loc-3-2) (adjacent loc-4-2 loc-4-1) (adjacent loc-4-2 loc-4-3)
    (adjacent loc-4-3 loc-3-3) (adjacent loc-4-3 loc-4-2) (adjacent loc-4-3 loc-4-4)
    (adjacent loc-4-4 loc-3-4) (adjacent loc-4-4 loc-4-3)
  )
  (:goal (and 
    (at num1 loc-1-1) (at num2 loc-1-2) (at num3 loc-1-3) (at num4 loc-1-4)
    (at num5 loc-2-1) (at num6 loc-2-2) (at num7 loc-2-3) (at num8 loc-2-4)
    (at num9 loc-3-1) (at num10 loc-3-2) (at num11 loc-3-3) (at num12 loc-3-4)
    (at num13 loc-4-1) (at num14 loc-4-2) (at num15 loc-4-3) (at B loc-4-4)
  ))
)
