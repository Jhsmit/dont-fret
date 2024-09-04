
<template>
    <v-treeview 
        :items="items" 
        dense 
        hoverable
        activatable
        v-model="value" 
        @update:active="value => active = value"
        :open="open_"
        @update:open="value => open_ = value"
        >
        
        <template v-slot:prepend="{ item }">
            <v-icon v-if="item.icon">{{ item.icon }}</v-icon>
        </template>
    </v-treeview>
</template>
