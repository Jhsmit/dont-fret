<template>
    <v-select
      v-model="value"
      :items="items"
      :label="label"
      :item_name="item_name"
      multiple
    >
      <template v-slot:selection="{ index }">
        
        <span v-if="index === 0">
            <span class="text--secondary">
            Selected {{item_name}}{{ value.length !== 1 ? 's' : '' }}:
            </span>
            <template v-if="value.length === items.length">
            all
            </template>
            <template v-else>
            {{ value.length }}
            </template>
        </span>
      </template>
    </v-select>
  </template>