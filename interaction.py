import math


class VirtualObject:

    def __init__(
        self,
        x,
        y,
        radius=60,
        name="Object"
    ):

        self.x = x

        self.y = y

        self.radius = radius

        self.name = name


class InteractionEngine:

    def __init__(self):

        self.objects = []

        self.grabbed_object = None


    # =========================================================
    # ADD OBJECT
    # =========================================================

    def add_object(self, obj):

        self.objects.append(obj)


    # =========================================================
    # FIND OBJECT UNDER GAZE
    # =========================================================

    def find_object(
        self,
        x,
        y
    ):

        closest = None

        closest_distance = float("inf")


        for obj in self.objects:

            distance = math.sqrt(
                (x - obj.x) ** 2
                +
                (y - obj.y) ** 2
            )


            if distance <= obj.radius:

                if distance < closest_distance:

                    closest = obj

                    closest_distance = distance


        return closest


    # =========================================================
    # GRAB
    # =========================================================

    def grab(
        self,
        obj
    ):

        if obj is None:

            return


        self.grabbed_object = obj


        print(
            f"GRABBED: {obj.name}"
        )


    # =========================================================
    # RELEASE
    # =========================================================

    def release(self):

        if self.grabbed_object:

            print(
                f"RELEASED: "
                f"{self.grabbed_object.name}"
            )


        self.grabbed_object = None


    # =========================================================
    # MOVE
    # =========================================================

    def move(
        self,
        x,
        y
    ):

        if self.grabbed_object:

            self.grabbed_object.x = x

            self.grabbed_object.y = y