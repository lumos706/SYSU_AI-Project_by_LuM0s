(define (domain spare_tire)
  (:requirements :strips:equality:typing:disjunctive-preconditions:negative-preconditions)
  (:types physob location) 
  (:predicates  (Tire ?x - physob)
		        (At ?x - physob ?y - location))
            
  (:constants
    Ground Axle - location
    Flat - physob   
  )
		
  (:action Remove
             :parameters (?x - physob ?y - location)
             :precondition (At ?x ?y)
             :effect (and (not (At ?x ?y)) (At ?x Ground))
  )

  (:action PutOn
             :parameters (?x - physob)
             :precondition (and (Tire ?x) (At ?x Ground) (not (At Flat Axle)))
             :effect (and (not (At ?x Ground)) (At ?x Axle))
  )
  ; (:action LeaveOvernight
  ;            :effect (and (not (At Spare Ground))(not (At Spare Axle))(not (At Spare Trunk))(not (At Flat Ground))(not (At Flat Axle))(not (At Flat Trunk)))
  ; )
 )
