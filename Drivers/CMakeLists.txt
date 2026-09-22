# Common BSP Interfaces (Header-only library)
add_library(stm32-bsp-common INTERFACE)

target_include_directories(stm32-bsp-common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/stm32-bsp-common
)

# B-U585I-IOT02A
add_library(b-u585i-iot02a-bsp STATIC
    b-u585i-iot02a-bsp/b_u585i_iot02a.c
    b-u585i-iot02a-bsp/b_u585i_iot02a_audio.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_bus.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_camera.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_eeprom.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_env_sensors.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_light_sensor.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_motion_sensors.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_ospi.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_ranging_sensor.c
    #b-u585i-iot02a-bsp/b_u585i_iot02a_usbpd_pwr.c
    b-u585i-iot02a-bsp/b_u585i_iot02a.h
    b-u585i-iot02a-bsp/b_u585i_iot02a_audio.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_bus.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_camera.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_conf_template.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_eeprom.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_env_sensors.h
    b-u585i-iot02a-bsp/b_u585i_iot02a_errno.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_light_sensor.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_motion_sensors.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_ospi.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_ranging_sensor.h
    #b-u585i-iot02a-bsp/b_u585i_iot02a_usbpd_pwr.h
)

target_link_libraries(b-u585i-iot02a-bsp PRIVATE
    STM32_Drivers
    stm32-bsp-common
)

target_include_directories(b-u585i-iot02a-bsp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/b-u585i-iot02a-bsp
)
