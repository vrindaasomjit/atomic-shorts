```python
from manim import *
import numpy as np

class StoryboardScene(MovingCameraScene):
    def construct(self):
        # --- Configuration ---
        # Define a consistent color palette
        BLUE_H = BLUE_E  # Hydrogen
        RED_O = RED_B    # Oxygen / Carbon
        YELLOW_E = YELLOW_B # Electrons
        TEAL_ACCENT = TEAL_C
        GREY_LINE = GREY_B

        # --- Scene 1: What is DFT? (Super Simple!) ---
        # Narration: "Have you ever wondered how scientists figure out what makes things tick, like why a diamond is super hard or why a banana ripens?"
        scene1_duration = 4

        title = Text("What is DFT?", font_size=72, color=TEAL_ACCENT).shift(UP*0.5)
        subtitle = Text("(Super Simple!)", font_size=48, color=BLUE_H).next_to(title, DOWN)
        
        # Abstract atoms floating around
        atom1 = Circle(radius=0.3, color=BLUE_H, fill_opacity=0.7).move_to(2 * LEFT + 2 * UP)
        atom2 = Circle(radius=0.4, color=RED_O, fill_opacity=0.7).move_to(3 * RIGHT + 1 * DOWN)
        atom3 = Circle(radius=0.25, color=YELLOW_E, fill_opacity=0.7).move_to(1 * LEFT + 1.5 * DOWN)
        atom4 = Circle(radius=0.35, color=TEAL_ACCENT, fill_opacity=0.7).move_to(2.5 * RIGHT + 2 * UP)
        
        atoms_floating = VGroup(atom1, atom2, atom3, atom4)

        self.play(
            FadeIn(title, shift=UP),
            FadeIn(subtitle, shift=DOWN),
            LaggedStart(*[FadeIn(atom.shift(np.random.rand(3)*0.5 - 0.25)) for atom in atoms_floating], lag_ratio=0.2)
        )
        self.wait(scene1_duration - 2) # Wait for narration
        self.play(
            FadeOut(title, shift=UP),
            FadeOut(subtitle, shift=DOWN),
            FadeOut(atoms_floating)
        )

        # --- Scene 2: Building Blocks of Everything ---
        # Narration: "Everything around us, from your toys to the air you breathe, is made of tiny, tiny building blocks called atoms. Atoms stick together to make molecules, like two hydrogen atoms and one oxygen atom making a water molecule!"
        scene2_duration = 5

        # Individual atoms
        h1 = Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).shift(LEFT * 2.5 + UP * 0.5)
        h2 = Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).shift(RIGHT * 2.5 + UP * 0.5)
        o_atom = Circle(radius=0.4, color=RED_O, fill_opacity=0.8).shift(DOWN * 0.5)
        
        h1_label = Text("H", font_size=30, color=BLACK).move_to(h1)
        h2_label = Text("H", font_size=30, color=BLACK).move_to(h2)
        o_label = Text("O", font_size=30, color=BLACK).move_to(o_atom)

        atoms_group = VGroup(h1, h2, o_atom, h1_label, h2_label, o_label)

        self.play(
            FadeIn(atoms_group, scale=0.5)
        )
        self.wait(scene2_duration / 2)

        # Form water molecule
        water_h1_pos = LEFT * 0.8 + DOWN * 0.5
        water_h2_pos = RIGHT * 0.8 + DOWN * 0.5
        water_o_pos = UP * 0.5

        bond1 = Line(water_o_pos, water_h1_pos, color=GREY_LINE, stroke_width=5)
        bond2 = Line(water_o_pos, water_h2_pos, color=GREY_LINE, stroke_width=5)
        
        self.play(
            h1.animate.move_to(water_h1_pos),
            h2.animate.move_to(water_h2_pos),
            o_atom.animate.move_to(water_o_pos),
            h1_label.animate.move_to(water_h1_pos),
            h2_label.animate.move_to(water_h2_pos),
            o_label.animate.move_to(water_o_pos),
            Create(bond1),
            Create(bond2),
            run_time=1.5
        )
        water_molecule = VGroup(h1, h2, o_atom, h1_label, h2_label, o_label, bond1, bond2)
        self.wait(scene2_duration / 2 - 1.5)
        self.play(FadeOut(water_molecule))

        # --- Scene 3: The Invisible Glue: Electrons ---
        # Narration: "What makes these atoms stick together? It's even tinier, super-fast particles called electrons! Think of them like invisible, super-sticky glue that holds the building blocks together."
        scene3_duration = 5

        # Recreate water molecule for zoom
        h1_zoom = Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(water_h1_pos)
        h2_zoom = Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(water_h2_pos)
        o_atom_zoom = Circle(radius=0.4, color=RED_O, fill_opacity=0.8).move_to(water_o_pos)
        bond1_zoom = Line(water_o_pos, water_h1_pos, color=GREY_LINE, stroke_width=5)
        bond2_zoom = Line(water_o_pos, water_h2_pos, color=GREY_LINE, stroke_width=5)
        water_molecule_zoom = VGroup(h1_zoom, h2_zoom, o_atom_zoom, bond1_zoom, bond2_zoom)

        self.play(FadeIn(water_molecule_zoom))
        self.play(self.camera.frame.animate.scale(0.7).move_to(water_molecule_zoom.get_center()), run_time=1)

        # Electrons zipping around
        electrons = VGroup()
        num_electrons = 10
        for _ in range(num_electrons):
            electron = Dot(radius=0.08, color=YELLOW_E, fill_opacity=1)
            electrons.add(electron)

        # Define paths for electrons (not directly used in the animation loop, but good for context)
        paths = []
        for i in range(num_electrons):
            start_point = water_molecule_zoom.get_center() + np.random.uniform(-1, 1, 3) * 0.5
            end_point = water_molecule_zoom.get_center() + np.random.uniform(-1, 1, 3) * 0.5
            paths.append(Line(start_point, end_point))

        if electrons: # Guard against empty group
            self.play(
                LaggedStart(*[FadeIn(e) for e in electrons], lag_ratio=0.1),
                run_time=0.5
            )
        
        # Animate electrons moving
        electron_animations = []
        for i, electron in enumerate(electrons):
            # Random path around the molecule
            path_points = [
                electron.get_center(),
                water_molecule_zoom.get_center() + np.random.uniform(-1, 1, 3) * 0.7,
                water_molecule_zoom.get_center() + np.random.uniform(-1, 1, 3) * 0.7,
                electron.get_center() # Loop back
            ]
            path = VMobject()
            path.set_points_as_corners(path_points)
            electron_animations.append(MoveAlongPath(electron, path, run_time=2, rate_func=linear, loop=True))

        if electron_animations: # Guard against empty list
            self.play(
                AnimationGroup(*electron_animations),
                run_time=scene3_duration - 1.5
            )
        self.play(
            FadeOut(water_molecule_zoom),
            FadeOut(electrons),
            self.camera.frame.animate.scale(1/0.7).move_to(ORIGIN), # Zoom out
            run_time=1
        )

        # --- Scene 4: Too Many Workers! ---
        # Narration: "Now, imagine you have a really big Lego castle. It has hundreds of tiny workers (electrons) running around, each doing something different. It would be super hard to watch every single worker at the same time, right?"
        scene4_duration = 5

        # Methane molecule (CH4)
        c_atom = Circle(radius=0.4, color=RED_O, fill_opacity=0.8).move_to(ORIGIN)
        h_atoms = VGroup(
            Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(UP + LEFT),
            Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(UP + RIGHT),
            Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(DOWN + LEFT),
            Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(DOWN + RIGHT)
        )
        bonds_ch4 = VGroup(
            Line(c_atom.get_center(), h_atoms[0].get_center(), color=GREY_LINE, stroke_width=5),
            Line(c_atom.get_center(), h_atoms[1].get_center(), color=GREY_LINE, stroke_width=5),
            Line(c_atom.get_center(), h_atoms[2].get_center(), color=GREY_LINE, stroke_width=5),
            Line(c_atom.get_center(), h_atoms[3].get_center(), color=GREY_LINE, stroke_width=5)
        )
        methane_molecule = VGroup(c_atom, h_atoms, bonds_ch4).scale(0.8)

        self.play(FadeIn(methane_molecule))

        # Many electrons zipping around
        many_electrons = VGroup()
        num_many_electrons = 50
        for _ in range(num_many_electrons):
            electron = Dot(radius=0.06, color=YELLOW_E, fill_opacity=1)
            many_electrons.add(electron)

        if many_electrons: # Guard against empty group
            self.play(
                LaggedStart(*[FadeIn(e.move_to(methane_molecule.get_center() + np.random.uniform(-1.5, 1.5, 3))) for e in many_electrons], lag_ratio=0.01),
                run_time=1
            )

        electron_animations_many = []
        for electron in many_electrons:
            path_points = [
                electron.get_center(),
                methane_molecule.get_center() + np.random.uniform(-1.5, 1.5, 3),
                methane_molecule.get_center() + np.random.uniform(-1.5, 1.5, 3),
                electron.get_center()
            ]
            path = VMobject()
            path.set_points_as_corners(path_points)
            electron_animations_many.append(MoveAlongPath(electron, path, run_time=3, rate_func=linear, loop=True))

        if electron_animations_many: # Guard against empty list
            self.play(
                AnimationGroup(*electron_animations_many),
                run_time=scene4_duration - 2
            )

        # Thought bubble with question mark
        thought_bubble = ThoughtBubble(
            direction=UL,
            width=3,
            height=2.5,
            fill_opacity=0.8,
            stroke_width=3,
            stroke_color=GREY_LINE
        ).next_to(methane_molecule, UP * 2 + RIGHT * 2)
        question_mark = Text("?", font_size=96, color=RED_O).move_to(thought_bubble.get_center())

        self.play(
            FadeIn(thought_bubble),
            FadeIn(question_mark)
        )
        self.wait(0.5)
        self.play(
            FadeOut(methane_molecule),
            FadeOut(many_electrons),
            FadeOut(thought_bubble),
            FadeOut(question_mark)
        )

        # --- Scene 5: The "Crowd Map" (Electron Density) ---
        # Narration: "Scientists had the same problem! So, instead of tracking every single electron, they came up with a clever idea. What if we just make a 'crowd map'? This map shows us where the electrons like to hang out the most, like a heatmap showing the busiest spots on a playground. We call this the 'electron density'."
        scene5_duration = 5

        # Recreate methane molecule
        methane_molecule_density = VGroup(
            Circle(radius=0.4, color=RED_O, fill_opacity=0.8).move_to(ORIGIN),
            VGroup(
                Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(UP + LEFT),
                Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(UP + RIGHT),
                Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(DOWN + LEFT),
                Circle(radius=0.3, color=BLUE_H, fill_opacity=0.8).move_to(DOWN + RIGHT)
            ),
            VGroup(
                Line(ORIGIN, UP + LEFT, color=GREY_LINE, stroke_width=5),
                Line(ORIGIN, UP + RIGHT, color=GREY_LINE, stroke_width=5),
                Line(ORIGIN, DOWN + LEFT, color=GREY_LINE, stroke_width=5),
                Line(ORIGIN, DOWN + RIGHT, color=GREY_LINE, stroke_width=5)
            )
        ).scale(0.8)
        self.play(FadeIn(methane_molecule_density))

        # Electron density cloud
        # Approximate with multiple overlapping ellipses/circles
        density_cloud_center = Ellipse(width=3.5, height=3.5, color=BLUE_H, fill_opacity=0.2, stroke_opacity=0)
        density_cloud_arms = VGroup(
            Ellipse(width=1.5, height=1.5, color=BLUE_H, fill_opacity=0.3, stroke_opacity=0).move_to(UP + LEFT).shift(UP*0.2+LEFT*0.2),
            Ellipse(width=1.5, height=1.5, color=BLUE_H, fill_opacity=0.3, stroke_opacity=0).move_to(UP + RIGHT).shift(UP*0.2+RIGHT*0.2),
            Ellipse(width=1.5, height=1.5, color=BLUE_H, fill_opacity=0.3, stroke_opacity=0).move_to(DOWN + LEFT).shift(DOWN*0.2+LEFT*0.2),
            Ellipse(width=1.5, height=1.5, color=BLUE_H, fill_opacity=0.3, stroke_opacity=0).move_to(DOWN + RIGHT).shift(DOWN*0.2+RIGHT*0.2)
        )
        electron_density_map = VGroup(density_cloud_center, density_cloud_arms).scale(0.8)

        self.play(
            FadeIn(electron_density_map, scale=0.5),
            run_time=2
        )
        
        density_label = Text("Electron Density", font_size=40, color=TEAL_ACCENT).next_to(electron_density_map, DOWN*2)
        self.play(FadeIn(density_label, shift=UP))
        self.wait(scene5_duration - 2)
        self.play(
            FadeOut(methane_molecule_density),
            FadeOut(electron_density_map),
            FadeOut(density_label)
        )

        # --- Scene 6: The Magic Shortcut: DFT! ---
        # Narration: "Here's the amazing part: a special science trick called 'Density Functional Theory' (or DFT for short) says that if you have this 'crowd map' of electrons, you can figure out almost *everything* about the molecule! You don't need to know what each individual electron is doing. Just the map tells you how strong the molecule is, what color it might be, or how it will react with other molecules!"
        scene6_duration = 6

        # Recreate electron density map
        electron_density_map_dft = VGroup(density_cloud_center.copy(), density_cloud_arms.copy()).scale(0.8)
        self.play(FadeIn(electron_density_map_dft))

        # Pulsing glow (Indicate is the modern replacement for ShowPassingFlash)
        self.play(
            Indicate(electron_density_map_dft, scale_factor=1.1, color=YELLOW_E),
            run_time=1.5
        )

        # Text bubbles for properties
        prop_strength = Text("Strength", font_size=36, color=BLUE_H).shift(UP*2 + LEFT*3)
        prop_color = Text("Color", font_size=36, color=RED_O).shift(UP*2 + RIGHT*3)
        prop_react = Text("How it reacts", font_size=36, color=TEAL_ACCENT).shift(DOWN*2 + LEFT*3)
        prop_magnetism = Text("Magnetism", font_size=36, color=YELLOW_E).shift(DOWN*2 + RIGHT*3)

        properties = VGroup(prop_strength, prop_color, prop_react, prop_magnetism)

        self.play(
            LaggedStart(
                FadeIn(prop_strength, shift=UP),
                FadeIn(prop_color, shift=UP),
                FadeIn(prop_react, shift=DOWN),
                FadeIn(prop_magnetism, shift=DOWN),
                lag_ratio=0.2
            ),
            run_time=2
        )

        # Magical sparkle animation
        sparkles = VGroup()
        for _ in range(20):
            sparkle = Dot(radius=0.05, color=YELLOW_E, fill_opacity=1).move_to(electron_density_map_dft.get_center() + np.random.uniform(-1.5, 1.5, 3))
            sparkles.add(sparkle)
        
        if sparkles: # Guard against empty group
            self.play(
                LaggedStart(*[FadeIn(s).animate.shift(np.random.uniform(-0.5, 0.5, 3)) for s in sparkles], lag_ratio=0.05),
                FadeOut(sparkles, scale=0.5),
                run_time=1
            )

        self.wait(scene6_duration - 1.5 - 2 - 1)
        self.play(
            FadeOut(electron_density_map_dft),
            FadeOut(properties)
        )

        # --- Scene 7: Why It's Super Useful! ---
        # Narration: "This 'magic shortcut' helps scientists design new medicines, create better batteries for your toys, invent super-strong materials, and even understand how plants grow! It's like having a secret decoder ring for the universe's building blocks!"
        scene7_duration = 5

        # Briefly show electron density