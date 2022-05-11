# Set list of examples
set(EXAMPLES
    "forward.cc"
    )

add_executable(examples_forward "forward.cc")
target_link_libraries(examples_forward nntile)

add_executable(examples_mixer "mixer.cc")
target_link_libraries(examples_mixer nntile)

