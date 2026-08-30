import math


class VirtualObject:

    def __init__(
        self,
        x,
        y,
        radius=50
    ):

        self.x = x
        self.y = y

        self.radius = radius


class InteractionEngine:

    def __init__(self):

        self.objects = []

        self.grabbed_object = None


    # =====================================
    # ADD OBJECT
    # =====================================

    def add_object(self, obj):

        self.objects.append(obj)


    # =====================================
    # FIND OBJECT
    # =====================================

    def find_object(
        self,
        x,
        y
    ):

        for obj in self.objects:

            distance = math.sqrt(

                (x - obj.x) ** 2

                +

                (y - obj.y) ** 2

            )


            if distance <= obj.radius:

                return obj


        return None


    # =====================================
    # PROCESS GESTURE
    # =====================================

    def process_event(
        self,
        event,
        x,
        y
    ):

        # ---------------------------------
        # START GRAB
        # ---------------------------------

        if event == "PINCH_START":

            obj = self.find_object(
                x,
                y
            )


            if obj:

                self.grabbed_object = obj

                print(
                    "GRABBED OBJECT"
                )


        # ---------------------------------
        # RELEASE
        # ---------------------------------

        elif event == "PINCH_END":

            if self.grabbed_object:

                print(
                    "RELEASED OBJECT"
                )

                self.grabbed_object = None


    # =====================================
    # UPDATE
    # =====================================

    def update(
        self,
        x,
        y,
        is_pinching
    ):

        if (
            is_pinching
            and
            self.grabbed_object
        ):

            self.grabbed_object.x = x

            self.grabbed_object.y = y