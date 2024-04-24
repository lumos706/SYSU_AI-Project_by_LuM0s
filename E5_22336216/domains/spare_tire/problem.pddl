(define (problem prob)
 (:domain spare_tire)
 (:objects  Spare - physob  Trunk  - location)
 
 (:init 
  (Tire Flat)
  (Tire Spare)
  (At Flat Axle)
  (At Spare Trunk)

 )
 (:goal
  (At Spare Axle)
 )
)
