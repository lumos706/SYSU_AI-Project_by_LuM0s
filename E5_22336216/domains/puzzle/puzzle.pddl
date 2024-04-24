(define (domain puzzle)
  (:requirements :strips :equality :typing)
  (:types num loc) 
  (:constants B - num) ; B represents the empty space
  (:predicates  
    (at ?n - num ?l - loc) ; ?n is at location ?l
    (adjacent ?l1 ?l2 - loc) ; ?l1 and ?l2 are adjacent locations
  )

  (:action slide
    :parameters (?n - num ?from - loc ?to - loc)
    :precondition (and (at ?n ?from) (at B ?to) (adjacent ?from ?to))
    :effect (and 
              (not (at ?n ?from)) 
              (at ?n ?to) 
              (not (at B ?to)) 
              (at B ?from)
            )
  )
)